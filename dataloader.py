from itertools import groupby
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import glob
import codecs
import os
import torch
import youtokentome

class SequenceLoader(object):
    def __init__(self, tokenizer, data_files, tokens_in_batch, for_training=False, pad_to_length=None):
        self.tokens_in_batch = tokens_in_batch
        self.pad_to_length = pad_to_length

        # Is this for training?
        self.for_training = for_training

        # Load BPE model
        self.tokenizer = tokenizer

        # Load data

        source_data = []
        target_data = []

        src_langs = []
        tgt_langs = []

        src_tgt_pairs = {}
        for data_file in data_files:
            file_name, single = data_file.split(".")
            split, pair = file_name.split("_")
            src, tgt = pair.split("-")

            for src_tgt_pair in src_tgt_pairs:
                if pair not in src_tgt_pair:
                    src_tgt_pairs[pair] = [None, None]

                src_tgt_pairs[pair]['src' if src == pair else 'tgt'] = src_tgt_pair

            """
            data structure of src_tgt_pairs:
            {
                'en-de': {
                    'src': 'train_en-de.en',
                    'tgt': 'train_en-de.de'
                },
                'de-en': {
                    'src': 'train_de-en.de',
                    'tgt': 'train_de-en.en'
                },
                ...
            }
            """

        for lang_pair, data_file_dict in src_tgt_pairs.items():
            with codecs.open(data_file_dict['src'], "r", encoding="utf-8") as f:
                src_langs.append(lang_pair.split("-")[0])
                source_data.append(f.read().split("\n")[:-1])

            with codecs.open(data_file_dict['tgt'], "r", encoding="utf-8") as f:
                tgt_langs.append(lang_pair.split("-")[1])
                target_data.append(f.read().split("\n")[:-1])

        assert len(source_data) == len(target_data), "There are a different number of source or target sequences!"

        source_lengths = [len(s) for s in tqdm(self.tokenizer.encode_all(source_data, bos=False, eos=False), desc='Encoding src sequences')]
        target_lengths = [len(t) for t in tqdm(self.tokenizer.encode_all(target_data, bos=True, eos=True), desc='Encoding tgt sequences')] # target language sequences have <BOS> and <EOS> tokens
        self.data = list(zip(source_data, target_data, source_lengths, target_lengths, src_langs, tgt_langs))

        # If for training, pre-sort by target lengths - required for itertools.groupby() later
        if self.for_training:
            self.data.sort(key=lambda x: x[3])

        # Create batches
        self.create_batches()

    def create_batches(self):
        if self.for_training:
            # Group or chunk based on target sequence lengths
            chunks = [list(g) for _, g in groupby(self.data, key=lambda x: x[3])]

            # Create batches, each with the same target sequence length
            self.all_batches = list()
            for chunk in chunks:
                # Sort inside chunk by source sequence lengths, so that a batch would also have similar source sequence lengths
                chunk.sort(key=lambda x: x[2])
                # How many sequences in each batch? Divide expected batch size (i.e. tokens) by target sequence length in this chunk
                seqs_per_batch = self.tokens_in_batch // chunk[0][3]
                # Split chunk into batches
                self.all_batches.extend([chunk[i: i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            # Shuffle batches
            shuffle(self.all_batches)
            self.n_batches = len(self.all_batches)
            self.current_batch = -1
        else:
            # Simply return once pair at a time
            self.all_batches = [[d] for d in self.data]
            self.n_batches = len(self.all_batches)
            self.current_batch = -1

    def __iter__(self):
        return self

    def __next__(self):
        # Update current batch index
        self.current_batch += 1
        try:
            source_data, target_data, source_lengths, target_lengths, src_langs, tgt_langs = zip(*self.all_batches[self.current_batch])
        # Stop iteration once all batches are iterated through
        except IndexError:
            raise StopIteration

        # Tokenize using BPE model to word IDs
        source_data = self.tokenizer.encode_all(source_data, src_langs, output_type=youtokentome.OutputType.ID, bos=False, eos=False)
        target_data = self.tokenizer.encode_all(target_data, tgt_langs, output_type=youtokentome.OutputType.ID, bos=True, eos=True)

        # Convert source and target sequences as padded tensors
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data], batch_first=True, padding_value=0)
        target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data], batch_first=True, padding_value=0)

        if self.pad_to_length is not None:
            source_data = torch.cat([source_data, torch.zeros(source_data.size(0), self.pad_to_length - source_data.size(1), dtype=source_data.dtype)], dim=1)
            target_data = torch.cat([target_data, torch.zeros(target_data.size(0), self.pad_to_length - target_data.size(1), dtype=target_data.dtype)], dim=1)

        # Convert lengths to tensors
        source_lengths = torch.LongTensor(source_lengths)
        target_lengths = torch.LongTensor(target_lengths)

        return source_data, target_data, source_lengths, target_lengths, src_langs, tgt_langs
