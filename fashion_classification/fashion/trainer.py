"""ModelTrainer: the one training/evaluation class used for every model."""
import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from torch.utils.tensorboard import SummaryWriter

from .data import denormalize


def compute_map(labels, probs, num_classes):
    """Mean Average Precision, computed from predicted class probabilities via one-vs-rest precision-recall curves."""
    y_true = label_binarize(labels, classes=list(range(num_classes))) # converts 1D target label into a 2D matrix
    # need this matrix when nworking with probability distribution across multiple classes
    return average_precision_score(y_true, probs, average="macro")
    # probs will be  2Darray of predicited class probabilities from softmax
    # 'macro' calaculates average precision for each class and computes the unweighted mean across all


class ModelTrainer:
    def __init__(self, model, optimizer, criterion, class_names, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device

    def train_one_epoch(self, loader):

        self.model.train() # sets model to train mode
        running_loss, correct, total = 0.0, 0, 0
        #iterates through batches
        for xb, yb in loader: #xb = input image, yb=labels
            xb, yb = xb.to(self.device), yb.to(self.device) #move data to GPU
            self.optimizer.zero_grad() #clear grad before adding new grad
            out = self.model(xb)
            loss = self.criterion(out, yb) #loss function
            loss.backward() #backpropagation to see how much each param contributed to the error
            self.optimizer.step() #update weights
  #track the loss and accuracy
            running_loss += loss.item() * xb.size(0) #size= the number of images in the batch
            correct += (out.argmax(1) == yb).sum().item()#argmax(1) finds the position of the largest score for each image
            total += xb.size(0)
        return running_loss / total, correct / total

    @torch.no_grad() # don't update grad
    def evaluate(self, loader, collect_probs=False): #helper for full_evalutaion
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            out = self.model(xb)
            loss = self.criterion(out, yb)

            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            # moves tnsors to the CPU because it converts to a normal python list, or numpy array
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
            if collect_probs: # to get the probability, softmax
                all_probs.append(torch.softmax(out, dim=1).cpu().numpy())

        val_loss, val_acc = running_loss / total, correct / total
        if collect_probs:
            return val_loss, val_acc, all_preds, all_labels, np.concatenate(all_probs, axis=0)
        return val_loss, val_acc, all_preds, all_labels

    def fit(self, train_loader, valid_loader, epochs, patience=5, log_dir=None,
            scheduler=None, verbose=True):
        """training loop with early stopping on validation accuracy.
        If log_dir is given, per-epoch loss/accuracy (and LR, if a scheduler
        is passed) are logged to TensorBoard."""
        writer = SummaryWriter(log_dir=log_dir) if log_dir else None #init tensorboard logger
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []} #to track per-epoch metrics
        best_val_acc = 0.0
        best_state = copy.deepcopy(self.model.state_dict()) #model.state_dict() returns references to the model's underlying parameter tensors
        epochs_no_improve = 0 #counter for early stop

        for epoch in range(1, epochs + 1):
            t0 = time.time() #record the time start
            tr_loss, tr_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(valid_loader)
            if scheduler is not None:
                scheduler.step(val_acc)
            dt = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

#tensorboard live logging
            if writer:
                writer.add_scalar("Loss/train", tr_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Accuracy/train", tr_acc, epoch)
                writer.add_scalar("Accuracy/val", val_acc, epoch)
                writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            if verbose: # print progress
                print(f"Epoch {epoch:2d}/{epochs} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                      f"val_loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.1f}s")

#early stop logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience: #5 times no improvement
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        self.model.load_state_dict(best_state)
        if writer:
            writer.flush()
            writer.close()
        return history, best_val_acc

    def fit_for_search(self, train_loader, valid_loader, epochs, trial):
        """loop used only during an Optuna trial for optimising hyperparams,
        reports validation accuracy each epoch so the trial can be pruned."""
        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            self.train_one_epoch(train_loader)
            # only needs the validation accuracy here, ignore the rest
            _, val_acc, _, _ = self.evaluate(valid_loader)
            best_val_acc = max(best_val_acc, val_acc)
            trial.report(val_acc, epoch)
            if trial.should_prune(): # should_prune() uses pruner defined in each model
                raise optuna.exceptions.TrialPruned()
        return best_val_acc

    def full_evaluation(self, valid_loader, title):
        """classification report + confusion matrix + mean average
        precision on the validation set."""
        val_loss, val_acc, preds, labels, probs = self.evaluate(valid_loader, collect_probs=True)
        print(f"\n{title} — validation accuracy: {val_acc:.4f} (loss {val_loss:.4f})\n")
        #imported, it gets detailed classification performance (includes Precision, recall, f1,etc)
        print(classification_report(labels, preds, target_names=self.class_names, digits=3))

        cm = confusion_matrix(labels, preds) # compare true and predicted labels
        plt.figure(figsize=(7, 6))
        """annot=true shows the number inside the cells
        fmt='d' displays the values as int"""
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f"{title} — Confusion Matrix (validation set)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
  #mAP needs model's probabiity for each class ->average precision scores for each class
  #average across classes ->mAP
        mean_ap = compute_map(labels, probs, self.num_classes)
        print(f"{title} — Mean Average Precision (mAP, macro over classes): {mean_ap:.4f}")

        return val_acc, mean_ap

