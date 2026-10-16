"""
Each model has the same three steps:
    build_*_components(cfg, paths, device) -> dm, model, optimizer, criterion
    tune_*(paths, device, ...)             -> Optuna study
    train_final_*(best_cfg, paths, device) -> dm, trainer, history, best_val_acc
"""
import optuna
import torch.nn as nn
import torch.optim as optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .data import FashionDataManager
from .models import MLPClassifier, PretrainedVGG16, VGG16Scratch
from .trainer import ModelTrainer

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep trial logs readable

# pretrained weights were learned at 224x224, so this is fixed rather than tuned
PRETRAINED_IMG_SIZE = 224


def _make_criterion(cfg, dm, device):
    #Cross entrophy loss
    if cfg.get("use_class_weights", False):
        return nn.CrossEntropyLoss(weight=dm.class_weights().to(device))
    return nn.CrossEntropyLoss()


def _run_study(objective, n_trials, seed, n_startup_trials, n_warmup_steps):
    #pruning strategy
    pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    # TPE= tree-structured parzen estimator, It decides which hyperparameter combinations to try next based on previous trial's result
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed), pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# 1. MLP
def build_mlp_components(cfg, paths, device):
  # cfg is a configuration dictionary containing the hyperparameters for the MLP.
    dm = FashionDataManager(paths.train, paths.valid, paths.test, img_size=cfg["img_size"],
                             batch_size=cfg["batch_size"], augment=cfg.get("augment", False))
    #data manager that handles loading training nad validation images, processing/resizing etc
    input_dim = 3 * cfg["img_size"] * cfg["img_size"]
    model = MLPClassifier(input_dim, cfg["hidden_layers"], dm.num_classes,
                           dropout=cfg["dropout"]).to(device)
    # Adam optimiser update the model's weights after calculating gradients
    #weight decay is a form of regularization that discourages very large weights
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    criterion = _make_criterion(cfg, dm, device)
    return dm, model, optimizer, criterion


def make_mlp_objective(paths, device, search_epochs=10):
    def mlp_objective(trial):
        img_size = trial.suggest_categorical("img_size", [64, 96, 128]) # diff sizes
        n_layers = trial.suggest_int("n_layers", 1, 3) # between 1 and 3
        # every hidden layers have between 32 and 512 neurons
        # searching on a logarithmic scale ensures equal, unbiased exploration of small, medium, and large values
        hidden_layers = [trial.suggest_int(f"n_units_l{i}", 32, 512, log=True) for i in range(n_layers)]
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])

        cfg = dict(img_size=img_size, hidden_layers=hidden_layers, dropout=dropout, lr=lr,
                   weight_decay=weight_decay, batch_size=batch_size,
                   use_class_weights=use_class_weights)
        #build model, it creates data manager, MLP, Adam, CrossEntropyLoss using the hyper params selected by Optuna
        dm, model, optimizer, criterion = build_mlp_components(cfg, paths, device)
        trainer = ModelTrainer(model, optimizer, criterion, dm.class_names, device)
        #train with 10 epoch
        val_acc = trainer.fit_for_search(dm.train_loader(), dm.valid_loader(),
                                         epochs=search_epochs, trial=trial)
        trial.set_user_attr("config", cfg) # an Optuna function used to attach custom metadata or arbitrary data to a specific trial.
        return val_acc
    return mlp_objective


def tune_mlp(paths, device, seed=42, n_trials=25, search_epochs=10):
    # first 4 runns without median-based pruning, and even after that, a trial gets at least the first 3 reported epochs before pruning can occur
    return _run_study(make_mlp_objective(paths, device, search_epochs), n_trials, seed,
                      n_startup_trials=4, n_warmup_steps=3)


def train_final_mlp(best_cfg, paths, device, epochs=40, patience=5, log_dir="runs/mlp"):
    dm, model, optimizer, criterion = build_mlp_components(best_cfg, paths, device)
    trainer = ModelTrainer(model, optimizer, criterion, dm.class_names, device)
    history, best_val_acc = trainer.fit(dm.train_loader(), dm.valid_loader(),
                                        epochs=epochs, patience=patience, log_dir=log_dir)
    return dm, trainer, history, best_val_acc


# 2. VGG16 from scratch
def build_vgg_scratch_components(cfg, paths, device):
    dm = FashionDataManager(paths.train, paths.valid, paths.test, img_size=cfg["img_size"],
                             batch_size=cfg["batch_size"], augment=cfg.get("augment", False))
    # creates the custom VGG16 model
    #img_size needs to be dividable by 32
    model = VGG16Scratch(num_classes=dm.num_classes, in_size=cfg["img_size"],
                          use_batchnorm=cfg["use_batchnorm"], dropout=cfg["dropout"]).to(device)

    if cfg["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    else:
      #anything other than Adam, it uses SGD
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg.get("weight_decay", 0.0))

    criterion = _make_criterion(cfg, dm, device)
    return dm, model, optimizer, criterion


def make_vgg_scratch_objective(paths, device, search_epochs=6):
    def vgg_scratch_objective(trial):
        img_size = trial.suggest_categorical("img_size", [128, 224])
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        augment = trial.suggest_categorical("augment", [True, False])
        use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])

        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        if optimizer_name == "adam":
            lr = trial.suggest_float("lr_adam", 1e-5, 1e-3, log=True)
        else:
            lr = trial.suggest_float("lr_sgd", 1e-3, 1e-1, log=True)

        cfg = dict(img_size=img_size, use_batchnorm=use_batchnorm, dropout=dropout,
                   optimizer=optimizer_name, lr=lr, weight_decay=weight_decay,
                   batch_size=batch_size, augment=augment, use_class_weights=use_class_weights)

        dm, model, optimizer, criterion = build_vgg_scratch_components(cfg, paths, device)
        trainer = ModelTrainer(model, optimizer, criterion, dm.class_names, device)
        val_acc = trainer.fit_for_search(dm.train_loader(), dm.valid_loader(),
                                         epochs=search_epochs, trial=trial)
        trial.set_user_attr("config", cfg)
        return val_acc
    return vgg_scratch_objective


def tune_vgg_scratch(paths, device, seed=42, n_trials=12, search_epochs=6):
    return _run_study(make_vgg_scratch_objective(paths, device, search_epochs), n_trials, seed,
                      n_startup_trials=2, n_warmup_steps=2)


def train_final_vgg_scratch(best_cfg, paths, device, epochs=30, patience=7,
                            log_dir="runs/vgg_scratch"):
    dm, model, optimizer, criterion = build_vgg_scratch_components(best_cfg, paths, device)
    trainer = ModelTrainer(model, optimizer, criterion, dm.class_names, device)
    #ReduceLROnPlateau is a dynamic learning rate scheduler in PyTorch that monitors
    # a specified validation metric (like loss or accuracy) during training
    # 'max' reduces LR when the quantity monitored has stopped increasing
    # factor multiplies by which the learning rate is reduced (new lr=old LR *factor)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    history, best_val_acc = trainer.fit(dm.train_loader(), dm.valid_loader(), epochs=epochs,
                                        patience=patience, log_dir=log_dir, scheduler=scheduler)
    return dm, trainer, history, best_val_acc


# 3. Pretrained VGG16
def build_pretrained_vgg_components(cfg, paths, device):
    dm = FashionDataManager(paths.train, paths.valid, paths.test, img_size=PRETRAINED_IMG_SIZE,
                             batch_size=cfg["batch_size"], augment=cfg.get("augment", False))
    model = PretrainedVGG16(num_classes=dm.num_classes, freeze_backbone=cfg["freeze_backbone"],
                             dropout=cfg["dropout"]).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "adam":
        optimizer = optim.Adam(trainable_params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    else:
        optimizer = optim.SGD(trainable_params, lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg.get("weight_decay", 0.0))

    criterion = _make_criterion(cfg, dm, device)
    return dm, model, optimizer, criterion


def make_pretrained_vgg_objective(paths, device, search_epochs=5):
    def pretrained_vgg_objective(trial):
        freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        augment = trial.suggest_categorical("augment", [True, False])
        use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

        # Fine-tuning the whole network needs a much smaller LR than training
        # just the head, or the pretrained weights get wrecked in a few steps.
        if freeze_backbone:
            lr = trial.suggest_float("lr_head", 1e-4, 1e-2, log=True)
        else:
            lr = trial.suggest_float("lr_finetune", 1e-6, 1e-4, log=True)

        cfg = dict(freeze_backbone=freeze_backbone, dropout=dropout, weight_decay=weight_decay,
                   batch_size=batch_size, augment=augment, use_class_weights=use_class_weights,
                   optimizer=optimizer_name, lr=lr)

        dm, model, optimizer, criterion = build_pretrained_vgg_components(cfg, paths, device)
        trainer = ModelTrainer(model, optimizer, criterion, dm.class_names, device)
        val_acc = trainer.fit_for_search(dm.train_loader(), dm.valid_loader(),
                                         epochs=search_epochs, trial=trial)
        trial.set_user_attr("config", cfg)
        return val_acc
    return pretrained_vgg_objective


def tune_pretrained_vgg(paths, device, seed=42, n_trials=10, search_epochs=5):
    return _run_study(make_pretrained_vgg_objective(paths, device, search_epochs), n_trials, seed,
                      n_startup_trials=3, n_warmup_steps=2)


def train_final_pretrained_vgg(best_cfg, paths, device, epochs=20, patience=5,
                               log_dir="runs/vgg_pretrained"):
    dm, model, optimizer, criterion = build_pretrained_vgg_components(best_cfg, paths, device)
    trainer = ModelTrainer(model, optimizer, criterion, dm.class_names, device)
    history, best_val_acc = trainer.fit(dm.train_loader(), dm.valid_loader(), epochs=epochs,
                                        patience=patience, log_dir=log_dir)
    return dm, trainer, history, best_val_acc
