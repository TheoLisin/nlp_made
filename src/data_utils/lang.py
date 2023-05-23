import string
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union
from torchtext.vocab import vocab, Vocab
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.api import TokenizerI


PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
TOKENS = [UNK, PAD, SOS, EOS]


class Lang(object):
    def __init__(self, name: str, tokenizer: TokenizerI):
        self.name = name
        self._vocab: Optional[Vocab] = None
        self._tok = tokenizer
        self._counter = Counter()

    def add_sentence(self, sentence: str) -> str:
        sent = self._tok.tokenize(sentence.lower())
        sent = [w for w in sent if w not in string.punctuation]
        self._counter.update(sent)
        return sent
    
    def create_vocab(self):
        if len(self._counter) == 0:
            raise ValueError("No words was added.")
        self._vocab = vocab(self._counter, min_freq=2, specials=TOKENS, special_first=True)
        self._vocab.set_default_index(self._vocab[UNK])
    
    def encode(self, tok_sent: List[str], pad_len: Optional[int] = None):
        toks = [SOS] + tok_sent + [EOS]
        if pad_len is not None and len(toks) < pad_len:
            diff = pad_len - len(toks)
            toks.extend([PAD for _ in range(diff)])
        
        return [self._vocab[tok] for tok in toks]

    def encode_raw(self, sent: str, pad_len: Optional[int] = None):
        toks = self._tok.tokenize(sent.lower())
        return self.encode(toks, pad_len)

    def decode(self, tokens: List[int]):
        seq = []
        for tid in tokens:
            tok = self._vocab.get_itos()[tid]
            if tok in [SOS, PAD, UNK]:
                continue
            elif tok == EOS:
                return seq
            seq.append(tok)
        return seq

    @property
    def vocab(self):
        if self._vocab is None:
            raise AttributeError("Vocab was not created.")
        return self._vocab
    
    def __repr__(self) -> str:
        return self.name


def read_langs(lang1: str, lang2: str, file_or_lines: Union[str, Path, List[str]]):
    # Read the file and split into lines
    if isinstance(file_or_lines, list):
        lines = file_or_lines
    else:
        with open(file_or_lines, "r", encoding='utf-8') as flines:
            lines = flines.readlines()
    
    lv1 = Lang(lang1, WordPunctTokenizer())
    lv2 = Lang(lang2, WordPunctTokenizer())
    pairs = []

    for line in lines:
        lng1_sent, lng2_sent = line.strip().split('\t')
        lng1_toks = lv1.add_sentence(lng1_sent)
        lng2_toks = lv2.add_sentence(lng2_sent)
        pairs.append(
            {
                lang1: lng1_toks,
                lang2: lng2_toks,
            }
        )

    lv1.create_vocab()
    lv2.create_vocab()

    return lv1, lv2, pairs
