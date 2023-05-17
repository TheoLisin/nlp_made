import pytorch_lightning as pl
from typing import Any, Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data_utils.dataset import TranslationDataset
from data_utils.lang import PAD


class PlTranslationDataset(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: TranslationDataset,
        val_dataset: TranslationDataset,
        train_bs: int,
        test_bs: int,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.src_pad = self.train_dataset.source_lang[PAD]
        self.trg_pad = self.train_dataset.target_lang[PAD]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_bs,
            num_workers=4,
            collate_fn=self.collate_batch,
        )

    def collate_batch(self, batch: Tuple[Tensor, Tensor]):
        src, trg = batch
        src_pad = pad_sequence(src, padding_value=self.src_pad)
        trg_pad = pad_sequence(trg, padding_value=self.trg_pad)
        return src_pad, trg_pad
