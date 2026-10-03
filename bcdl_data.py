"""
Data loading for the bcdl-faces dataset.

Expects:
    bcdl-faces/
        001.jpg ... 060.jpg
        annotations.csv      (columns: imagename,person)
"""

import csv
import os
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --- Step 1: CSV -> list of (filename, label_index) ---------------------------

def read_annotations(csv_path, classes):
    samples = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            # .index() raises ValueError if the name doesn't match CLASSES exactly
            samples.append((row['imagename'].strip(), classes.index(row['person'].strip())))
    return samples


# --- Step 2: Dataset -- turns one (filename, label) into (tensor, label) ------

class FacesDataset(Dataset):
    def __init__(self, root, samples, transform):
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, label = self.samples[i]
        img = Image.open(os.path.join(self.root, name)).convert('RGB')
        return self.transform(img), label


# --- Step 3: Train/validation split, same number of images per actor ---------

def split_per_class(samples, val_per_class, seed=0):
    rng = random.Random(seed)            # fixed seed -> same split every run
    train, val = [], []
    for label in sorted({lbl for _, lbl in samples}):
        items = [s for s in samples if s[1] == label]
        rng.shuffle(items)
        val += items[:val_per_class]
        train += items[val_per_class:]
    return train, val


# --- Step 4: Transforms (augmentation only for training) ---------------------

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # random crop of 80-100% of the image
    transforms.RandomHorizontalFlip(),                     # a mirrored face is the same person
    transforms.ColorJitter(0.2, 0.2, 0.2),                 # small brightness/contrast/saturation changes
    transforms.ToTensor(),                                 # PIL [H,W,3] 0..255 -> tensor [3,H,W] 0..1
    transforms.Normalize(MEAN, STD),                       # 0..1 -> -1..1
])

val_tf = transforms.Compose([
    transforms.Resize(224),          # shorter side -> 224, keeps aspect ratio
    transforms.CenterCrop(224),      # -> 224x224
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# --- Step 5: DataLoaders -- group samples into batches ------------------------

def make_loaders(root, classes, batch_size=8, val_per_class=6, num_workers=2):
    samples = read_annotations(os.path.join(root, 'annotations.csv'), classes)
    train_s, val_s = split_per_class(samples, val_per_class)

    train_dl = DataLoader(FacesDataset(root, train_s, train_tf),
                          batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(FacesDataset(root, val_s, val_tf),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl


if __name__ == '__main__':
    classes = ('Al Pacino', 'Robert De Niro')
    train_dl, val_dl = make_loaders('bcdl-faces', classes)

    print(len(train_dl.dataset), len(val_dl.dataset))   # 48 12
    x, y = next(iter(train_dl))
    print(x.shape, x.dtype)                              # torch.Size([8, 3, 224, 224]) torch.float32
    print(y)                                             # tensor of 8 labels, 0s and 1s
