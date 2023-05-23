import pytorch_lightning as pl
from typing import Any, Tuple, List
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
        test_dataset: TranslationDataset,
        train_bs: int,
        test_bs: int,
    ):
        super().__init__()
        self.target_lang = train_dataset.target_lang
        self.source_lang = train_dataset.source_lang
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.src_pad = self.train_dataset.source_lang.vocab[PAD]
        self.trg_pad = self.train_dataset.target_lang.vocab[PAD]

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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_bs,
            num_workers=4,
            collate_fn=self.collate_batch,
        )

    def collate_batch(self, batch: List[Tuple[Tensor, Tensor]]):
        src = [src_b[0] for src_b in batch]
        trg = [trg_b[1] for trg_b in batch]

        src_pad = pad_sequence(src, padding_value=self.src_pad)
        trg_pad = pad_sequence(trg, padding_value=self.trg_pad)
        return src_pad, trg_pad
