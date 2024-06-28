from criteria.labelsmooth import LabelSmoothedCE
from modules import transformer
from prettytable import PrettyTable
from tokenizers import Tokenizer
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataloader
import glob
import io
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import utils

args, unk = utils.get_args()

setattr(args, 'device', torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"))
setattr(args, 'dtype', torch.float32 if args.dtype == 'float32' else torch.float16)

run_dir = os.path.join('runs', args.run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

steps = 0

def forward_pass(epoch, src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths, model, scaler, criterion):
    src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
    tgt_key_padding_mask = tgt_seqs == 0 # (N, max_target_sequence_pad_length_this_batch)

    if scaler is not None:
        with autocast():
            predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)
    else:
        predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)

    moe_diversity_loss = 0
    encoder_moe_gating_variances = None
    decoder_moe_gating_variances = None

    if args.moe_diversity_loss_coefficient > 0 and epoch >= args.moe_diversity_inclusion_epoch:
        encoder_moe_gating_variances = torch.stack(encoder_moe_gating_variances).std(dim=0).mean()
        decoder_moe_gating_variances = torch.stack(decoder_moe_gating_variances).std(dim=0).mean()

        moe_diversity_loss = ((encoder_moe_gating_variances + decoder_moe_gating_variances) / 2) * args.moe_diversity_loss_coefficient

        encoder_moe_gating_variances = encoder_moe_gating_variances.item()
        decoder_moe_gating_variances = decoder_moe_gating_variances.item()

    # Note: If the target sequence is "<bos> w1 w2 ... wN <eos> <PAD> <PAD> <PAD> <PAD> ..."
    # we should consider only "w1 w2 ... wN <eos>" as <BOS> is not predicted
    # Therefore, pads start after (length - 1) positions
    if scaler is not None:
        with autocast():
            translation_loss = criterion(inputs=predicted_sequences, targets=tgt_seqs, lengths=tgt_seq_lengths)
    else:
        translation_loss = criterion(inputs=predicted_sequences, targets=tgt_seqs, lengths=tgt_seq_lengths) # scalar

    return translation_loss, moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances

def viz_attn_weights(stack_name, layer_num, n_head, activation_weights, attendee_tokens, attending_tokens):
    fig, ax = plt.subplots(figsize=(10, 10))
    s = sns.heatmap(activation_weights, square=True, annot=True, annot_kws={"fontsize":6}, fmt=".4f", xticklabels=attendee_tokens, yticklabels=attending_tokens, ax=ax)
    s.set(xlabel="Attending Tokens", ylabel="Attended Tokens", title=f"{stack_name}-Attn Layer {layer_num} Head {n_head} Weights")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return buf

def viz_model(model, tokenizer, summary_writer, src, tgt):
    model = utils.sanitize_model(model)

    if not tgt.endswith('<eos>'):
        tgt += '<eos>'

    with torch.no_grad():
        model.eval()

        input_sequence = torch.LongTensor(tokenizer.encode(src, add_special_tokens=True).ids).unsqueeze(0).to(args.device)
        input_tokens = [utils.clean_decoded_text(tokenizer.decode([id.item()], skip_special_tokens=False)) for id in input_sequence.squeeze(0)]
        input_sequence_length = input_sequence.size(1)

        # pad input sequence to args.maxlen
        if args.use_infinite_attention or True:
            input_sequence = torch.cat([input_sequence, torch.zeros([1, args.maxlen - input_sequence.size(1)], dtype=torch.long, device=input_sequence.device)], dim=1)

        target_sequence = torch.LongTensor(tokenizer.encode(tgt, add_special_tokens=True).ids).unsqueeze(0).to(args.device)
        target_tokens = [utils.clean_decoded_text(tokenizer.decode([id.item()], skip_special_tokens=False)) for id in target_sequence.squeeze(0)]
        target_sequence_length = target_sequence.size(1)

        # pad target sequence to args.maxlen
        if args.use_infinite_attention or True:
            target_sequence = torch.cat([target_sequence, torch.zeros([1, args.maxlen - target_sequence.size(1)], dtype=torch.long, device=target_sequence.device)], dim=1)

        src_key_padding_mask = input_sequence == 0 # (N, pad_length)
        tgt_key_padding_mask = target_sequence == 0 # (N, pad_length)

        input_sequence = model.encoder.perform_embedding_transformation(input_sequence) # (N, pad_length, d_model)
        input_sequence = model.encoder.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)

        for e, encoder_layer in enumerate(model.encoder.encoder_layers):
            input_sequence, attention_weights = encoder_layer.self_attn(input_sequence, input_sequence, input_sequence, src_key_padding_mask)

            for a, attention_weight_grid in enumerate(attention_weights):
                attention_weight_grid = attention_weight_grid.cpu().contiguous()

                # shape of attention_weights will be (1, n_heads, input_sequence_length, input_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                for i in range(attention_weight_grid.size(1)):
                    image_data = viz_attn_weights('Encoder-Self', e, i, attention_weight_grid[:, i, :input_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).cpu().detach().numpy(), input_tokens, input_tokens)
                    summary_writer.add_image(f"Encoder Layer {e} Head {i} Self-Attn Weights for Segment {a}", plt.imread(image_data), global_step=steps, dataformats='HWC')

            input_sequence, _ = encoder_layer.fcn(sequences=input_sequence) # (N, pad_length, d_model)

        input_sequence = model.encoder.norm(input_sequence)

        target_sequence = model.decoder.apply_embedding_transformation(target_sequence) # (N, pad_length, d_model)
        target_sequence = model.decoder.apply_positional_embedding(target_sequence) # (N, pad_length, d_model)

        for d, decoder_layer in enumerate(model.decoder.decoder_layers):
            target_sequence, attention_weights = decoder_layer.self_attn(target_sequence, target_sequence, target_sequence, tgt_key_padding_mask) # (N, pad_length, d_model)
            
            for a, attention_weight_grid in enumerate(attention_weights):
                attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                # shape of attention_weight_grid will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                for i in range(attention_weight_grid.size(1)):
                    image_data = viz_attn_weights('Decoder-Self', d, i, attention_weight_grid[:, i, :target_sequence_length, :target_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, target_tokens)
                    summary_writer.add_image(f"Decoder Layer {d} Head {i} Self-Attn Weights for Segment {a}", plt.imread(image_data), global_step=steps, dataformats='HWC')

            target_sequence, attention_weights = decoder_layer.cross_attn(target_sequence, input_sequence, input_sequence, src_key_padding_mask) # (N, pad_length, d_model)

            for a, attention_weight_grid in enumerate(attention_weights):
                attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
                for i in range(attention_weight_grid.size(1)):
                    image_data = viz_attn_weights('Decoder-Cross', d, i, attention_weight_grid[:, i, :target_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, input_tokens)
                    summary_writer.add_image(f"Decoder Layer {d} Head {i} Cross-Attn Weights for Segment {a}", plt.imread(image_data), global_step=steps, dataformats='HWC')

            target_sequence, _ = decoder_layer.fcn(target_sequence) # (N, pad_length, d_model)

def train_epoch(rank, model, epoch, train_loader, scaler, criterion, optimizer, summary_writer, tokenizer, early_stopping=None):
    global steps

    # training mode enables dropout
    model.train()

    step_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    translation_losses = utils.AverageMeter()
    encoder_moe_gating_variance_losses = utils.AverageMeter()
    decoder_moe_gating_variance_losses = utils.AverageMeter()

    start_step_time = time.time()

    for i, batch in enumerate(train_loader):
        if batch is None:
            break

        src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths = batch

        src_seqs = src_seqs.to(rank)
        tgt_seqs = tgt_seqs.to(rank)
        src_seq_lengths = src_seq_lengths.to(rank)
        tgt_seq_lengths = tgt_seq_lengths.to(rank)

        tgt_seq_length_sum = (tgt_seq_lengths - 1).sum().item()

        translation_loss, moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances = forward_pass(epoch, src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths, model, scaler, criterion)
        loss = translation_loss + moe_diversity_loss
        translation_losses.update(translation_loss.item(), tgt_seq_length_sum)
        total_losses.update(loss.item(), tgt_seq_length_sum)

        if encoder_moe_gating_variances is not None and decoder_moe_gating_variances is not None:
            encoder_moe_gating_variance_losses.update(encoder_moe_gating_variances, 1)
            decoder_moe_gating_variance_losses.update(decoder_moe_gating_variances, 1)

        # run again with src and tgt switched
        translation_loss, moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances = forward_pass(epoch, tgt_seqs, src_seqs, tgt_seq_lengths, src_seq_lengths, model, scaler, criterion)
        loss = loss + translation_loss + moe_diversity_loss
        translation_losses.update(translation_loss.item(), tgt_seq_length_sum)
        total_losses.update(loss.item(), tgt_seq_length_sum)

        if encoder_moe_gating_variances is not None and decoder_moe_gating_variances is not None:
            encoder_moe_gating_variance_losses.update(encoder_moe_gating_variances, 1)
            decoder_moe_gating_variance_losses.update(decoder_moe_gating_variances, 1)

        del src_seqs
        del src_seq_lengths
        del tgt_seqs
        del tgt_seq_lengths

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            (loss / args.batches_per_step).backward()

        # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
        if i % args.batches_per_step == 0:
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            steps += 1 # steps is a counter for the number of times the model has been updated via backprop

            utils.change_lr(optimizer, new_lr=utils.get_lr(steps, args.d_model, args.warmup_steps))

            step_time.update(time.time() - start_step_time)
            start_step_time = time.time()

            if rank == 0:
                if steps % args.print_frequency == 0:
                    print('\nEpoch {0}/{1}-----Batch {2}-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----Loss {total_losses.val:.4f} ({total_losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(epoch + 1, args.epochs, i + 1, step_time=step_time, total_losses=total_losses, early_stop_counter=early_stopping.counter if early_stopping is not None else 0, early_stop_patience=early_stopping.patience if early_stopping is not None else 0))
                    evaluate(model, tokenizer, summary_writer, src='<en>Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt_lang_code='de')
                    evaluate(model, tokenizer, summary_writer, src='<en>Anyone who retains the ability to recognise beauty will never become old.', tgt='Quiconque conserve la capacité de reconnaître la beauté ne vieillira jamais.', tgt_lang_code='fr')
                    evaluate(model, tokenizer, summary_writer, src='<de>Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt='Người nào giữ được khả năng nhận biết cái đẹp sẽ không bao giờ già.', tgt_lang_code='vi')

                summary_writer.add_scalar('Translation Training Loss', translation_losses.avg, steps)
                summary_writer.add_scalar('Training Loss', total_losses.avg, steps)
                if moe_diversity_loss > 0:
                    summary_writer.add_scalar('Encoder MoE Gating Variances', encoder_moe_gating_variance_losses.avg, steps)
                    summary_writer.add_scalar('Decoder MoE Gating Variances', decoder_moe_gating_variance_losses.avg, steps)

                # 'epoch' is 0-indexed
                # early stopping requires the ability to average the last few checkpoints so just save all of them
                if (epoch in [args.epochs - 1, args.epochs - 2] or args.early_stop) and steps % 1500 == 0:
                    utils.save_checkpoint(epoch, model, optimizer, prefix=f"{run_dir}/step{str(steps)}_")
        
def validate_epoch(rank, model, epoch, val_loader, scaler, criterion, summary_writer, tokenizer):
    model.eval()

    with torch.no_grad():
        losses = utils.AverageMeter()
        for batch in tqdm(val_loader):
            if batch is None:
                break

            src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths = batch

            src_seqs = src_seqs.to(rank)
            tgt_seqs = tgt_seqs.to(rank)
            src_seq_lengths = src_seq_lengths.to(rank)
            tgt_seq_lengths = tgt_seq_lengths.to(rank)

            tgt_seq_length_sum = (tgt_seq_lengths - 1).sum().item()

            src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
            tgt_key_padding_mask = tgt_seqs == 0 # (N, max_target_sequence_pad_length_this_batch)

            if scaler is not None:
                with autocast():
                    predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)
            else:
                predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)

            moe_diversity_loss = 0
            encoder_moe_gating_variances = None
            decoder_moe_gating_variances = None

            if args.moe_diversity_loss_coefficient > 0 and epoch >= args.moe_diversity_inclusion_epoch:
                encoder_moe_gating_variances = torch.stack(encoder_moe_gating_variances).std(dim=0).mean()
                decoder_moe_gating_variances = torch.stack(decoder_moe_gating_variances).std(dim=0).mean()

                moe_diversity_loss = ((encoder_moe_gating_variances + decoder_moe_gating_variances) / 2) * args.moe_diversity_loss_coefficient

                encoder_moe_gating_variances = encoder_moe_gating_variances.item()
                decoder_moe_gating_variances = decoder_moe_gating_variances.item()

            del src_seqs
            del src_seq_lengths

            # Note: If the target sequence is "<bos> w1 w2 ... wN <eos> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <eos>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            if scaler is not None:
                with autocast():
                    translation_loss = criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1)
            else:
                translation_loss = criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1) # scalar

            del tgt_seqs
            del tgt_seq_lengths

            loss = translation_loss + moe_diversity_loss

            losses.update(loss.item(), tgt_seq_length_sum)

        if rank == 0:
            summary_writer.add_scalar('Validation Loss', losses.avg, steps)
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            viz_model(model, tokenizer, summary_writer, "<en>Anyone who retains the ability to recognise beauty will never become old.", "<de>Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

        return losses.avg

def evaluate(model, tokenizer, summary_writer, src, tgt, tgt_lang_code):
    global steps

    best, _ = utils.beam_search_translate(args, src, model, tokenizer, tgt_lang_code, device=args.device, beam_size=4, length_norm_coefficient=0.6)

    debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
    debug_validate_table.add_row([src, best, tgt])

    console_size = os.get_terminal_size()
    debug_validate_table.max_width = (console_size.columns // 3) - 15
    debug_validate_table.min_width = (console_size.columns // 3) - 15

    print(debug_validate_table)
    with open(f"{run_dir}/debug_validate.html", "a") as f:
        f.write(debug_validate_table.get_html_string() + "<br>")

    summary_writer.add_text(f'Validation/{tgt_lang_code}', best, steps)

def load_model(args, rank, tokenizer):
    model = transformer.Transformer(args, tokenizer.get_vocab_size())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    start_epoch = 0
    if rank == 0:
        if os.path.exists(os.path.join(run_dir, 'transformer_checkpoint.pth.tar')):
            start_epoch = utils.load_checkpoint(model, optimizer, os.path.join(run_dir, 'transformer_checkpoint.pth.tar'))
        utils.save_checkpoint(start_epoch, model, optimizer, f"runs/{args.run_name}/")

    dist.barrier()

    start_epoch = utils.load_checkpoint(model, optimizer, os.path.join(run_dir, 'transformer_checkpoint.pth.tar'))

    if args.torch_compile_model:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = int(args.dynamo_cache_size_limit)

        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    model = model.to(rank)

    model = DDP(model, device_ids=[rank])

    return start_epoch, model, optimizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # You can use any free port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    summary_writer = SummaryWriter(log_dir=run_dir)

    tokenizer = Tokenizer.from_file("tokenizers/tokenizer_collated.json")

    start_epoch, model, optimizer = load_model(args, rank, tokenizer)

    criterion = LabelSmoothedCE(args, rank, eps=args.label_smoothing).to(args.device)

    if args.early_stop:
        early_stopping = utils.EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
    else:
        early_stopping = None

    if rank == 0:
        utils.print_model(model)
        print(f"Optimizer: {optimizer}")
        print(f"Criterion: {criterion}")

    if 'use_amp' in args and args.use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    try:
        # if start_epoch == 0 and rank == 0:
        #     print("Visualizing attention weights before training...")
        #     # get attention weight visualization before any updates are made to the model
        #     with torch.no_grad():
        #         model.eval()
        #         viz_model(model, tokenizer, summary_writer, "<en>Anyone who retains the ability to recognise beauty will never become old.", "<de>Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

        print(f"Training for {args.epochs} epochs...")
        start = time.time()

        for epoch in range(start_epoch, args.epochs):
            for n in tqdm(range(args.n_files), desc=f"Epoch {epoch + 1}/{args.epochs}"):
                train_loader = dataloader.get_generator(n, tokenizer, 'data', 'train', args.tokens_in_batch)(rank)
                val_loader = dataloader.get_generator(0, tokenizer, 'data', 'validation', args.tokens_in_batch)(rank)

                train_epoch(rank, model, epoch, train_loader, scaler, criterion, optimizer, summary_writer, tokenizer, early_stopping)

                val_loss_avg = validate_epoch(rank, model, epoch, val_loader, scaler, criterion, summary_writer, tokenizer)

                utils.save_checkpoint(epoch, model, optimizer, prefix=run_dir)

                if rank == 0:
                    if early_stopping is not None:
                        if early_stopping(val_loss_avg):
                            print("Early stopping")
                            utils.average_checkpoints(args.epochs, optimizer, run_dir, args.early_stop_num_latest_checkpoints_to_avg, model_name_prefix='step')
                            break

        time_taken = time.time() - start

        if rank == 0:
            print(f"Training complete. Averaging checkpoints...")
            utils.average_checkpoints(args.epochs, optimizer, run_dir, model_name_prefix='step')

            test_loader = dataloader.get_generator(0, tokenizer, 'data', 'validation', args.tokens_in_batch, rank)

            print("Training complete. Scoring with sacrebleu...")
            print(utils.sacrebleu_evaluate(args, run_dir, tokenizer, model, device=args.device, sacrebleu_in_python=True, test_loader=test_loader).score, time_taken, utils.count_parameters(model))
    finally:
        cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
