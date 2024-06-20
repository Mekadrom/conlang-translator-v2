from itertools import groupby
from random import shuffle
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import codecs
import os
import torch
import youtokentome

class SequenceLoader(object):
    def __init__(self, args, src_tokenizer, tgt_tokenizer, data_folder, source_suffix, target_suffix, split, tokens_in_batch, pad_to_length=None):
        self.args = args
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        self.tokens_in_batch = tokens_in_batch
        self.pad_to_length = pad_to_length

        self.split = split.lower()

        # Is this for training?
        self.for_training = self.split.startswith("train")

        # Load BPE model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        # Load data
        with codecs.open(os.path.join(data_folder, ".".join([split, source_suffix])), "r", encoding="utf-8") as f:
            source_data = f.read().split("\n")[:-1]
        with codecs.open(os.path.join(data_folder, ".".join([split, target_suffix])), "r", encoding="utf-8") as f:
            target_data = f.read().split("\n")[:-1]
            target_data = [f"<bos>{t}<eos>" for t in target_data]

        assert len(source_data) == len(target_data), "There are a different number of source or target sequences!"

        if args.separate_tokenizers:
            source_lengths = [len(s) for s in tqdm(self.src_tokenizer.encode_all(source_data, bos=False, eos=False), desc='Encoding src sequences')]
            target_lengths = [len(t) for t in tqdm(self.tgt_tokenizer.encode_all(target_data, bos=True, eos=True), desc='Encoding tgt sequences')] # target language sequences have <bos> and <eos> tokens
        else:
            source_lengths = [len(s) for s in tqdm([self.src_tokenizer.encode(datum) for datum in source_data], desc='Encoding src sequences')]
            target_lengths = [len(t) for t in tqdm([self.tgt_tokenizer.encode(datum) for datum in target_data], desc='Encoding tgt sequences')] # target language sequences have <bos> and <eos> tokens
        self.data = list(zip(source_data, target_data, source_lengths, target_lengths))

        # If for training, pre-sort by target lengths - required for itertools.groupby() later
        if self.for_training:
            self.data.sort(key=lambda x: x[3])

        # Create batches
        self.create_batches()

    def create_batches(self):
        """
        Prepares batches for one epoch.
        """
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
        """
        Iterators require this method defined.
        """
        return self

    def __next__(self):
        """
        Iterators require this method defined.

        :returns: the next batch, containing:
            source language sequences, a tensor of size (N, encoder_sequence_pad_length)
            target language sequences, a tensor of size (N, decoder_sequence_pad_length)
            true source language lengths, a tensor of size (N)
            true target language lengths, typically the same as decoder_sequence_pad_length as these sequences are bucketed by length, a tensor of size (N)
        """
        # Update current batch index
        self.current_batch += 1
        try:
            source_data, target_data, source_lengths, target_lengths = zip(*self.all_batches[self.current_batch])
        # Stop iteration once all batches are iterated through
        except IndexError:
            raise StopIteration

        # Tokenize using BPE model to word IDs
        if self.args.separate_tokenizers:
            source_data = self.src_tokenizer.encode_all(source_data, output_type=youtokentome.OutputType.ID, bos=False, eos=False)
            target_data = self.tgt_tokenizer.encode_all(target_data, output_type=youtokentome.OutputType.ID, bos=True, eos=True)
        else:
            source_data = [self.src_tokenizer.encode(datum).ids for datum in source_data]
            target_data = [self.tgt_tokenizer.encode(datum).ids for datum in target_data]

        # Convert source and target sequences as padded tensors
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data], batch_first=True, padding_value=0)
        target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data], batch_first=True, padding_value=0)

        if self.pad_to_length is not None:
            if self.pad_to_length - source_data.size(1) > 0:
                source_data = torch.cat([source_data, torch.zeros(source_data.size(0), self.pad_to_length - source_data.size(1), dtype=source_data.dtype)], dim=1)
            elif self.pad_to_length - source_data.size(1) < 0:
                source_data = source_data[:, :self.pad_to_length+1]

            if self.pad_to_length - target_data.size(1) > 0:
                target_data = torch.cat([target_data, torch.zeros(target_data.size(0), self.pad_to_length - target_data.size(1), dtype=target_data.dtype)], dim=1)
            elif self.pad_to_length - target_data.size(1) < 0:
                target_data = target_data[:, :self.pad_to_length+1]

        # Convert lengths to tensors
        source_lengths = torch.LongTensor(source_lengths)
        target_lengths = torch.LongTensor(target_lengths)

        return source_data, target_data, source_lengths, target_lengths
