from typing import List, Tuple, Union
from torch import Tensor, tensor
from torch.utils.data import Dataset

from data_utils.lang import Lang, read_langs


class TranslationDataset(Dataset):
    def __init__(
        self,
        lines: List[str],
        source_lang: Union[str, Lang],
        target_lang: Union[str, Lang],
    ) -> None:
        """_summary_

        :param lines: list of lines in `source\ttarget` format
        :type lines: List[str]
        :param source_lang: name of lang or Lang with vocab for source
        :type source_lang: Union[str, Lang]
        :param target_lang: name of lang or Lang with vocab for target
        :type target_lang: Union[str, Lang]
        """
        
        if isinstance(source_lang, str):
            if not isinstance(target_lang, str):
                raise ValueError("source_lang and target_lang must be the same type")

            self.source_lang, self.target_lang, self.pairs = read_langs(
                source_lang, target_lang, lines
            )
            self.source_name = source_lang
            self.target_name = target_lang
        else:
            self.source_lang = source_lang
            self.target_lang = target_lang
            self.source_name = source_lang.name
            self.target_name = target_lang.name
            _, _, self.pairs = read_langs(self.source_name, self.target_name, lines)
        
        self.enc_pairs = self.encode_all()
    
    def encode_all(self):
        enc_pairs = []
        for pair in self.pairs:
            source = self.source_lang.encode(pair[self.source_name])
            target = self.target_lang.encode(pair[self.target_name])
            enc_pairs.append(
                {
                    self.source_name: source,
                    self.target_name: target,
                }
            )
        return enc_pairs

    def __len__(self):
        return len(self.enc_pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        source = self.enc_pairs[idx][self.source_name]
        target = self.enc_pairs[idx][self.target_name]
        return tensor(source), tensor(target)
