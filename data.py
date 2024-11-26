import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from pathlib import Path
from PIL import Image


class UADetracDataset(Dataset):
    """
    Custom Dataset for UA-DETRAC dataset with YOLO format labels.
    Images are expected to be sequential, with labels corresponding to each frame.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        self.label_paths = sorted(self.label_dir.glob("*.txt"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Load labels
        label_path = self.label_paths[idx]
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                class_id, cx, cy, w, h = map(float, line.strip().split())
                labels.append([class_id, cx, cy, w, h])

        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))

        return image, labels


class UADetracDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for UA-DETRAC dataset.
    Handles loading and splitting data for training, validation, and testing.
    """
    def __init__(self, dir, batch_size=4, num_workers=4, transform=None):
        super().__init__()
        self.directory = dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform or T.Compose([
            T.Resize((540, 960)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def setup(self, stage=None):
        """
        Split dataset into train, validation, and test sets.
        Assumes the dataset has already been pre-split by file structure:
        - train/: Training images and labels
        - val/: Validation images and labels
        - test/: Testing images and labels
        """
        if stage in (None, "fit"):
            self.train_dataset = UADetracDataset(
                image_dir=Path(self.directory) / "train/images",
                label_dir=Path(self.directory) / "train/labels",
                transform=self.transform,
            )
            self.val_dataset = UADetracDataset(
                image_dir=Path(self.directory) / "val/images",
                label_dir=Path(self.directory) / "val/labels",
                transform=self.transform,
            )
        if stage in (None, "test"):
            self.test_dataset = UADetracDataset(
                image_dir=Path(self.directory) / "val/images",
                label_dir=Path(self.directory) / "val/labels",
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length labels.
        Groups images and labels into batches.
        """
        images, labels = zip(*batch)
        images = torch.stack(images)
        return images, labels
