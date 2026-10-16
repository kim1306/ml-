"""Data handling: paths, the unlabeled-image dataset, and FashionDataManager.

FashionDataManager is the single data class reused by the MLP, the
from-scratch VGG16, and the pretrained VGG16.
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

#use imageNet's mean and std value for RGB channel
# mean and std values for image normalization in deep learning is below
# becuase VGG16 model that was pretrained on ImageNet
# imageNet = a massive image DB
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DataPaths:
    train: str
    valid: str
    test: str

    @classmethod
    def from_root(cls, root):
        return cls(
            train=os.path.join(root, "train"),
            valid=os.path.join(root, "valid"),
            test=os.path.join(root, "test"),
        )


class ImageDataset(Dataset):
    """Loads images from a folder for prediction,
    keeping track of filenames since there are no ground-truth labels."""
    def __init__(self, root, transform):
        self.paths = sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True))
        """recursively searches through a specified root directory and all of its
         subdirectories to build a sorted list of every file and folder path it finds.
         glob() used to search for files and folders on computer that match a specific pattern -
          ** with recursive=True tells the search to dive into every subfolde"""
        self.paths = [p for p in self.paths if p.lower().endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB") # this coverts other types of images (e.g. greyscale) to RGB for consistnecy
        return self.transform(img), os.path.basename(path)


class FashionDataManager:
    """The data-handling class, reused for every model in this
    notebook. Each instance is configured for one combination """

    def __init__(self, train_dir, valid_dir, test_dir, img_size, batch_size,
                 augment=False, num_workers=2):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.num_workers = num_workers

        self.train_tf, self.eval_tf = self._build_transforms()
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_tf)
        self.valid_dataset = datasets.ImageFolder(valid_dir, transform=self.eval_tf)

        self.class_names = self.train_dataset.classes
        self.num_classes = len(self.class_names)

    def _build_transforms(self):
        norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if self.augment:
            train_tf = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                # flipping, rotating, changing brightness is to avoid overfitting
                # instead of learning specific pixel locations, it should learn the actual pattern
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(), # converts integer pixel values from the [0, 255] range into floating-point numbers in the [0.0, 1.0] range.
                norm, #applies the pre-defined ImageNet normalization
            ])
        else:
            train_tf = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                norm,
            ])
        eval_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            norm,
        ])
        return train_tf, eval_tf

    def train_loader(self):
      # DataLoader automate and manage how data is delivered to the model
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                           num_workers=self.num_workers, pin_memory=True)

    def valid_loader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                           num_workers=self.num_workers, pin_memory=True)

    def test_loader(self, batch_size=8):
        test_dataset = ImageDataset(self.test_dir, self.eval_tf)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False), test_dataset

    def class_weights(self):
        """training dataset has a significant class imbalance (e.g. shorts': 207, 'tees': 1055)
        so this function calculates how much importance each class should get
        during training when some classes have more images than others"""
        counts = np.zeros(self.num_classes)
        for _, label in self.train_dataset.samples: #_ (underscore) means it doesn't care about the image path
        #label = index
        #.samples contains every trainning image
            counts[label] += 1
        # the number of entire images/ number of classs images * class counts
        # more images = smaller weight, fewer immages = larger weight
        weights = counts.sum() / (self.num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)


def denormalize(img_tensor):
    """Undoes ImageNet normalization so an image tensor can be displayed."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()
