from itertools import groupby
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import glob
import codecs
import os
import random
import torch
import youtokentome

class SequenceLoader(object):
    def __init__(self, tokenizer, data_files, tokens_in_batch, cache_count=1000, for_training=False, pad_to_length=None):
        self.data_files = data_files
        self.tokens_in_batch = tokens_in_batch
        self.cache_count = cache_count
        self.for_training = for_training
        self.pad_to_length = pad_to_length

        # Load BPE model
        self.tokenizer = tokenizer

        self.src_tgt_pairs = {}
        for data_file in self.data_files:
            print(f"data_file: {data_file}")
            file_name, single = data_file.split(".")
            split, pair = file_name.split("_")
            src, tgt = pair.split("-")

            if pair not in self.src_tgt_pairs.keys():
                self.src_tgt_pairs[pair] = {'src': None, 'tgt': None }

            self.src_tgt_pairs[pair]['src' if src == single else 'tgt'] = data_file

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

        print(f"src_tgt_pairs: {self.src_tgt_pairs}")

        self.load_data()

        # Create batches
        self.create_batches()

        print(f"n_batches: {self.n_batches} len(self.data): {len(self.data)}")

    def load_data(self):
        self.data = []

        source_data = []
        target_data = []

        src_langs = []
        tgt_langs = []

        for _ in range(self.cache_count):
            translation_pair = random.choice(list(self.src_tgt_pairs.keys()))
            src_file, tgt_file = self.src_tgt_pairs[translation_pair]['src'], self.src_tgt_pairs[translation_pair]['tgt']

            with codecs.open(src_file, "r", encoding="utf-8") as f:
                src_langs.append(src_file.split(".")[-1])
                source_data.append(f.read().split("\n")[:-1])

            with codecs.open(tgt_file, "r", encoding="utf-8") as f:
                tgt_langs.append(tgt_file.split(".")[-1])
                target_data.append(f.read().split("\n")[:-1])

        assert len(source_data) == len(target_data), "There are a different number of source or target sequences!"

        if len(source_data) == 0:
            raise ValueError("No more data to load!")

        source_lengths = [len(s) for s in tqdm(self.tokenizer.encode_all(source_data, langs=src_langs, bos=False, eos=False), desc='Encoding src sequences')]
        target_lengths = [len(t) for t in tqdm(self.tokenizer.encode_all(target_data, langs=tgt_langs, bos=True, eos=True), desc='Encoding tgt sequences')] # target language sequences have <BOS> and <EOS> tokens
        self.data = list(zip(source_data, target_data, source_lengths, target_lengths, src_langs, tgt_langs))

        # If for training, pre-sort by target lengths - required for itertools.groupby() later
        if self.for_training:
            self.data.sort(key=lambda x: x[3])

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
        except IndexError:
            # grab next cache set of data if no more batches
            try:
                self.load_data() # throws value error if no more lines
                self.create_batches()

                # reset batch index and return new batch
                self.current_batch = 0
                source_data, target_data, source_lengths, target_lengths, src_langs, tgt_langs = zip(*self.all_batches[self.current_batch])
            except ValueError:
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
