from torch.nn.utils.rnn import pad_sequence
from itertools import groupby
from random import shuffle
from tqdm import tqdm

import codecs
import os
import torch

def get_generator(n_file_idx, tokenizer, data_folder, split, tokens_in_batch):
    def load_data(rank):
        print(f"Loading {split}_{n_file_idx} on rank {rank}...")

        src_file_path = os.path.join(data_folder, f"{split}_{n_file_idx}.src")
        tgt_file_path = os.path.join(data_folder, f"{split}_{n_file_idx}.tgt")

        with codecs.open(src_file_path, "r", encoding="utf-8") as src_file, codecs.open(tgt_file_path, "r", encoding="utf-8") as tgt_file:
            while True:
                batch = []
                total_tokens = 0
                for src_line, tgt_line in zip(src_file, tgt_file):
                    src_seq = tokenizer.encode(src_line[:-1], add_special_tokens=True).ids
                    tgt_seq = tokenizer.encode(tgt_line[:-1], add_special_tokens=True).ids

                    src_length = len(src_seq)
                    tgt_length = len(tgt_seq)
                    
                    batch.append((torch.LongTensor(src_seq), torch.LongTensor(tgt_seq), 
                                  torch.LongTensor([src_length]), torch.LongTensor([tgt_length])))
                    
                    total_tokens += tgt_length
                    
                    if total_tokens >= tokens_in_batch:
                        break
                
                if len(batch) == 0:
                    return  # End of file reached
                
                src_seqs, tgt_seqs, src_lengths, tgt_lengths = zip(*batch)
                
                src_seqs = pad_sequence(src_seqs, batch_first=True, padding_value=0)
                src_lengths = torch.cat(src_lengths)
                tgt_seqs = pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
                tgt_lengths = torch.cat(tgt_lengths)
                
                yield src_seqs, tgt_seqs, src_lengths, tgt_lengths

    return load_data

class SequenceLoader(object):
    def __init__(self, args, src_tokenizer, tgt_tokenizer, data_folder, src_file_name, tgt_file_name, tokens_in_batch, pad_to_length=None, for_training=True):
        self.args = args
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.tokens_in_batch = tokens_in_batch
        self.pad_to_length = pad_to_length
        self.for_training = for_training

        print(f"Loading {src_file_name} and {tgt_file_name}...")

        # Load data
        with codecs.open(os.path.join(data_folder, src_file_name), "r", encoding="utf-8") as f:
            source_data = f.read().split("\n")[:-1]

        with codecs.open(os.path.join(data_folder, tgt_file_name), "r", encoding="utf-8") as f:
            target_data = f.read().split("\n")[:-1]

        assert len(source_data) == len(target_data), "There are a different number of source or target sequences!"

        source_lengths = [len(s) for s in tqdm([self.src_tokenizer.encode(datum, add_special_tokens=True).ids for datum in source_data], desc='Encoding src sequences')] # source language sequences do not have <bos> and <eos> tokens
        target_lengths = [len(t) for t in tqdm([self.tgt_tokenizer.encode(datum, add_special_tokens=True).ids for datum in target_data], desc='Encoding tgt sequences')] # target language sequences have <eos> token but not <bos>, lang code tag serves that purpose

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
        source_data = [self.src_tokenizer.encode(datum, add_special_tokens=True).ids for datum in source_data]
        target_data = [self.tgt_tokenizer.encode(datum, add_special_tokens=True).ids for datum in target_data]

        # Convert source and target sequences as padded tensors
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data], batch_first=True, padding_value=0)
        target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data], batch_first=True, padding_value=0)

        if self.pad_to_length is not None:
            if self.pad_to_length - source_data.size(1) > 0:
                source_data = torch.cat([source_data, torch.zeros(source_data.size(0), self.pad_to_length - source_data.size(1), dtype=source_data.dtype)], dim=1)
            elif self.pad_to_length - source_data.size(1) < 0:
                source_data = source_data[:, :self.pad_to_length]
            if self.pad_to_length - target_data.size(1) > 0:
                target_data = torch.cat([target_data, torch.zeros(target_data.size(0), self.pad_to_length - target_data.size(1), dtype=target_data.dtype)], dim=1)
            elif self.pad_to_length - target_data.size(1) < 0:
                target_data = target_data[:, :self.pad_to_length]

        # Convert lengths to tensors
        source_lengths = torch.LongTensor(source_lengths)
        target_lengths = torch.LongTensor(target_lengths)

        return source_data, target_data, source_lengths, target_lengths
