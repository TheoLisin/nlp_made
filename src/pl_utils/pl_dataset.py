import pytorch_lightning as pl
from typing import Any
from torch.utils.data import DataLoader

from data_utils.dataset import TranslationDataset


class TranslationDataset(pl.LightningDataModule):
    def __init__(self, train_data: Any, val_data: Any, train_bs: int, test_bs: int):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_bs = train_bs
        self.test_bs = test_bs

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TranslationDataset(self.train_data)
            self.val_dataset = TranslationDataset(self.val_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_bs,
            num_workers=4,
        )
