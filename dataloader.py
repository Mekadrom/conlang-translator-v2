from itertools import groupby
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import codecs
import os
import torch

def generate_loader(n_file_idx, tokenizer, data_folder, split, tokens_in_batch, pad_to_length=None):
    def load_data(suffix):
        with codecs.open(os.path.join(data_folder, f"{split}_{n_file_idx}.{suffix}"), "r", encoding="utf-8") as f:
            for line in f:
                tokens = tokenizer.encode(line, add_special_tokens=True).ids
                yield torch.LongTensor(tokens), torch.LongTensor([len(tokens)])

    src_data = load_data("src")
    tgt_data = load_data("tgt")

    def make_next_batch():
        batch = []
        total_tokens = 0
        while total_tokens < tokens_in_batch:
            try:
                src, src_lengths = next(src_data)
                tgt, tgt_lengths = next(tgt_data)
            except StopIteration:
                break
            batch.append((src, tgt, src_lengths, tgt_lengths))

            example_length = tgt_lengths.item()

            # batching metric is # of tokens in target sequences
            total_tokens += example_length

            if len(batch) == 0:
                return None
        
        src, tgt, src_lengths, tgt_lengths, = zip(*batch)

        # takes a list of id sequences and returns padded Tensor
        src = pad_sequence(sequences=src, batch_first=True, padding_value=0)
        src_lengths = torch.stack(src_lengths).squeeze(-1)

        tgt = pad_sequence(sequences=tgt, batch_first=True, padding_value=0)
        tgt_lengths = torch.stack(tgt_lengths).squeeze(-1)

        return src, tgt, src_lengths, tgt_lengths

    yield from iter(make_next_batch, None)
