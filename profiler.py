from criteria.labelsmooth import LabelSmoothedCE
from modules import transformer
from torch.profiler import profile, ProfilerActivity

from tokenizers import Tokenizer

import dataloader
import torch
import torch.nn as nn
import torch.optim as optim
import utils

args, unk = utils.get_args()

setattr(args, 'device', torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"))
setattr(args, 'dtype', torch.float32 if args.dtype == 'float32' else torch.float16)

tokenizer = Tokenizer.from_file("tokenizers/tokenizer_collated.json")

model = transformer.Transformer(args, tokenizer.get_vocab_size())
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device=args.device, dtype=args.dtype, non_blocking=True)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

if args.torch_compile_model:
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = int(args.dynamo_cache_size_limit)
    compiled_model = torch.compile(model)
else:
    compiled_model = model

train_loader = dataloader.get_generator(0, tokenizer, 'data', 'train', args.tokens_in_batch, pad_to_length=args.maxlen)
val_loader = dataloader.get_generator(0, tokenizer, 'data', 'validation', args.tokens_in_batch, pad_to_length=args.maxlen)

criterion = LabelSmoothedCE(args, eps=args.label_smoothing).to(args.device)

def get_criteria(batch):
    src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths = batch

    src_seqs = src_seqs.to(args.device) # (1, max_source_sequence_pad_length_this_batch)
    tgt_seqs = tgt_seqs.to(args.device) # (1, max_target_sequence_pad_length_this_batch)
    src_seq_lengths = src_seq_lengths.to(args.device) # (1)
    tgt_seq_lengths = tgt_seq_lengths.to(args.device) # (1)

    src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
    tgt_key_padding_mask = tgt_seqs == 0 # (N, max_target_sequence_pad_length_this_batch)

    predicted_sequences, _, _ = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)

    moe_diversity_loss = 0

    del src_seqs
    del src_seq_lengths

    # Note: If the target sequence is "<bos> w1 w2 ... wN <eos> <PAD> <PAD> <PAD> <PAD> ..."
    # we should consider only "w1 w2 ... wN <eos>" as <BOS> is not predicted
    # Therefore, pads start after (length - 1) positions
    translation_loss = criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1) # scalar

    del tgt_seqs
    del tgt_seq_lengths

    loss = translation_loss + moe_diversity_loss

    return loss

steps = 0

print("Profiling...")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
    for i, batch in enumerate(train_loader):
        if i >= 20:
            break
        tgt_seq_length_sum = (batch[3] - 1).sum().item()

        loss = get_criteria(batch)

        (loss / args.batches_per_step).backward()

        # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
        if (i + 1) % args.batches_per_step == 0:
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            optimizer.step()

            optimizer.zero_grad()

            steps += 1

            utils.change_lr(optimizer, new_lr=utils.get_lr(steps, args.d_model, args.warmup_steps))

            print('Epoch {0}/{1}-----Batch {2}/{3}-----Step {4}/{5}-----'.format(1, 1, i + 1,  train_loader.n_batches, steps, args.n_steps))

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))
