from torch.nn.utils.rnn import pad_sequence

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
                    print(f"Rank {rank} processing {src_line[:-1]} and {tgt_line[:-1]}...")

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
