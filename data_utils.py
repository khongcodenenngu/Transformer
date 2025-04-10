import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# ---------------------------------------------------------------------------
# Special tokens
special_symbols = ['<pad>', '<unk>', '<bos>', '<eos>']
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# Tokenizers
token_transform = {
    'de': get_tokenizer('spacy', language='de_core_news_sm'),
    'en': get_tokenizer('spacy', language='en_core_web_sm')
}

def yield_tokens(data_iter, lang): # source - de, target - en
    for src, tgt in data_iter:
        text = src if lang == 'de' else tgt
        yield token_transform[lang](text.lower())

# Build vocab
def build_vocab():
    vocab_transform = {}
    for lang in ['de', 'en']:
        train_iter = Multi30k(split='train')
        vocab_transform[lang] = build_vocab_from_iterator(
            yield_tokens(train_iter, lang), specials=special_symbols
        )
        vocab_transform[lang].set_default_index(UNK_IDX)
    return vocab_transform

# ---------------------------------------------------------------------------
def tensor_transform(token_ids):
    return torch.tensor([BOS_IDX] + token_ids + [EOS_IDX], dtype=torch.long)

def collate_fn(batch, token_transform, vocab_transform):
    src_batch, tgt_batch = [], []

    for src_sample, tgt_sample in batch:
        src_tokens = token_transform['de'](src_sample.lower())
        tgt_tokens = token_transform['en'](tgt_sample.lower())

        src_ids = [vocab_transform['de'][token] for token in src_tokens]
        tgt_ids = [vocab_transform['en'][token] for token in tgt_tokens]

        src_tensor = tensor_transform(src_ids)
        tgt_tensor = tensor_transform(tgt_ids)

        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

    return src_batch, tgt_batch

# ---------------------------------------------------------------------------
"""
in src sentence - mask padding ids
in tgt sentence - mask padding ids + ahead ids (to prevent peeking ahead)
1 - keep, 0 - mask
"""
def create_src_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt, pad_idx):
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
    return tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(1)