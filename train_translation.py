from criteria.labelsmooth import LabelSmoothedCE
from modules import transformer
from prettytable import PrettyTable
from tokenizers import processors, Tokenizer
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataloader
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

args, unk = utils.get_args()

setattr(args, 'device', torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"))
setattr(args, 'dtype', torch.float32 if args.dtype == 'float32' else torch.float16)

run_dir = os.path.join('runs', args.run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

steps = 0

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

    with torch.no_grad():
        model.eval()

        src_seq = torch.LongTensor(tokenizer.encode(src, add_special_tokens=True).ids).unsqueeze(0).to(args.device)
        src_token_strs = [utils.clean_decoded_text(tokenizer.decode([id.item()], skip_special_tokens=False)) for id in src_seq.squeeze(0)]
        src_seq_len = src_seq.size(1)

        # pad input sequence to args.maxlen
        if args.use_infinite_attention or True:
            src_seq = torch.cat([src_seq, torch.zeros([1, args.maxlen - src_seq.size(1)], dtype=torch.long, device=src_seq.device)], dim=1)

        tgt_seq = torch.LongTensor(tokenizer.encode(tgt, add_special_tokens=True).ids).unsqueeze(0).to(args.device)
        tgt_token_strs = [utils.clean_decoded_text(tokenizer.decode([id.item()], skip_special_tokens=True)) for id in tgt_seq.squeeze(0)]
        tgt_seq_len = tgt_seq.size(1)

        # pad target sequence to args.maxlen
        if args.use_infinite_attention or True:
            tgt_seq = torch.cat([tgt_seq, torch.zeros([1, args.maxlen - tgt_seq.size(1)], dtype=torch.long, device=tgt_seq.device)], dim=1)

        src_key_padding_mask = src_seq == 0 # (N, pad_length)
        tgt_key_padding_mask = tgt_seq == 0 # (N, pad_length)

        src_seq = model.encoder.perform_embedding_transformation(src_seq) # (N, pad_length, d_model)
        src_seq = model.encoder.apply_positional_embedding(src_seq) # (N, pad_length, d_model)

        for e, encoder_layer in enumerate(model.encoder.encoder_layers):
            src_seq, attention_weights = encoder_layer.self_attn(src_seq, src_seq, src_seq, src_key_padding_mask)

            for a, attention_weight_grid in enumerate(attention_weights):
                attention_weight_grid = attention_weight_grid.cpu().contiguous()

                # shape of attention_weights will be (1, n_heads, input_sequence_length, input_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                for i in range(attention_weight_grid.size(1)):
                    image_data = viz_attn_weights('Encoder-Self', e, i, attention_weight_grid[:, i, :src_seq_len, :src_seq_len].transpose(-2, -1).squeeze(0).cpu().detach().numpy(), src_token_strs, src_token_strs)
                    summary_writer.add_image(f"Encoder Layer {e} Head {i} Self-Attn Weights for Segment {a}", plt.imread(image_data), global_step=steps, dataformats='HWC')

            src_seq, _ = encoder_layer.fcn(sequences=src_seq) # (N, pad_length, d_model)

        src_seq = model.encoder.norm(src_seq)

        tgt_seq = model.decoder.apply_embedding_transformation(tgt_seq) # (N, pad_length, d_model)
        tgt_seq = model.decoder.apply_positional_embedding(tgt_seq) # (N, pad_length, d_model)

        for d, decoder_layer in enumerate(model.decoder.decoder_layers):
            tgt_seq, attention_weights = decoder_layer.self_attn(tgt_seq, tgt_seq, tgt_seq, tgt_key_padding_mask) # (N, pad_length, d_model)
            
            for a, attention_weight_grid in enumerate(attention_weights):
                attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                # shape of attention_weight_grid will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                for i in range(attention_weight_grid.size(1)):
                    image_data = viz_attn_weights('Decoder-Self', d, i, attention_weight_grid[:, i, :tgt_seq_len, :tgt_seq_len].transpose(-2, -1).squeeze(0).numpy(), tgt_token_strs, tgt_token_strs)
                    summary_writer.add_image(f"Decoder Layer {d} Head {i} Self-Attn Weights for Segment {a}", plt.imread(image_data), global_step=steps, dataformats='HWC')

            tgt_seq, attention_weights = decoder_layer.cross_attn(tgt_seq, src_seq, src_seq, src_key_padding_mask) # (N, pad_length, d_model)

            for a, attention_weight_grid in enumerate(attention_weights):
                attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
                for i in range(attention_weight_grid.size(1)):
                    image_data = viz_attn_weights('Decoder-Cross', d, i, attention_weight_grid[:, i, :tgt_seq_len, :src_seq_len].transpose(-2, -1).squeeze(0).numpy(), tgt_token_strs, src_token_strs)
                    summary_writer.add_image(f"Decoder Layer {d} Head {i} Cross-Attn Weights for Segment {a}", plt.imread(image_data), global_step=steps, dataformats='HWC')

            tgt_seq, _ = decoder_layer.fcn(tgt_seq) # (N, pad_length, d_model)

def monitor_gradients_and_activations(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if np.isnan(grad_norm):
                print(f"Step {step}: NaN gradient in {name}")
            elif grad_norm > 1000:
                print(f"Step {step}: Large gradient in {name}: {grad_norm}")
        
        if torch.isnan(param).any():
            print(f"Step {step}: NaN parameter in {name}")

def forward_pass(rank, epoch, src_seqs, tgt_seqs, tgt_seq_lengths, model, scaler, criterion, tokenizer):
    src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
    tgt_key_padding_mask = tgt_seqs[:, :-1] == 0 # (N, max_target_sequence_pad_length_this_batch)

    if scaler is not None:
        with autocast():
            predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs[:, :-1], src_key_padding_mask, tgt_key_padding_mask)
    else:
        predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs[:, :-1], src_key_padding_mask, tgt_key_padding_mask)

    # if args.debug and rank == 0:
    if args.debug:
        print(f"src: {tokenizer.decode(src_seqs[0].tolist(), skip_special_tokens=False)}")
        print(f"predicted_sequences.max: {predicted_sequences[0].max()}")
        print(f"predicted_sequences.min: {predicted_sequences[0].min()}")
        print(f"predicted_sequences: {torch.argmax(F.softmax(predicted_sequences[0], dim=-1), dim=-1)}")
        print(f"predicted_sequences.shape: {predicted_sequences.shape}")
        print(f"src_seqs.shape: {src_seqs.shape}")
        print(f"src_seqs: {src_seqs[0]}")
        print(f"tgt_seqs.shape: {tgt_seqs.shape}")
        print(f"tgt_seqs: {tgt_seqs[0]}")
        print(f"tgt_seqs[:, 1:]: {tgt_seqs[0, 1:]}")
        print(f"tgt_seq_lengths: {tgt_seq_lengths[0]}")

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
            translation_loss = criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1)
    else:
        translation_loss = criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1)

    return translation_loss, moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances

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

        # if args.debug and rank == 0:
        if args.debug:
            print(f"src_seqs: {src_seqs.shape}")
            print(f"tgt_seqs: {tgt_seqs.shape}")
            print(f"src_seq_lengths: {src_seq_lengths.shape}")
            print(f"tgt_seq_lengths: {tgt_seq_lengths.shape}")

        src_seqs = src_seqs.to(rank)
        tgt_seqs = tgt_seqs.to(rank)
        src_seq_lengths = src_seq_lengths.to(rank)
        tgt_seq_lengths = tgt_seq_lengths.to(rank)

        tgt_seq_length_sum = (tgt_seq_lengths - 1).sum().item()

        translation_loss, moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances = forward_pass(rank, epoch, src_seqs, tgt_seqs, tgt_seq_lengths, model, scaler, criterion, tokenizer)
        loss = translation_loss + moe_diversity_loss
        translation_losses.update(translation_loss.item(), tgt_seq_length_sum)
        total_losses.update(loss.item(), tgt_seq_length_sum)

        if encoder_moe_gating_variances is not None and decoder_moe_gating_variances is not None:
            encoder_moe_gating_variance_losses.update(encoder_moe_gating_variances, 1)
            decoder_moe_gating_variance_losses.update(decoder_moe_gating_variances, 1)

        # run again with src and tgt switched
        translation_loss, moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances = forward_pass(rank, epoch, tgt_seqs, src_seqs, src_seq_lengths, model, scaler, criterion, tokenizer)
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
                # if args.debug and rank == 0:
                if args.debug:
                    monitor_gradients_and_activations(model, steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            steps += 1 # steps is a counter for the number of times the model has been updated via backprop

            new_lr = utils.get_lr(steps, args.d_model, args.warmup_steps)
            summary_writer.add_scalar('Learning Rate', new_lr, steps)
            utils.change_lr(optimizer, new_lr=new_lr)

            step_time.update(time.time() - start_step_time)
            start_step_time = time.time()

            # if rank == 0:
            if True:
                if steps % args.print_frequency == 0:
                    print('\nEpoch {0}/{1}-----Batch {2}-----Steps {3}-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----Loss {total_losses.val:.4f} ({total_losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(epoch + 1, args.epochs, i + 1, steps, step_time=step_time, total_losses=total_losses, early_stop_counter=early_stopping.counter if early_stopping is not None else 0, early_stop_patience=early_stopping.patience if early_stopping is not None else 0))

                    evaluate(model, tokenizer, summary_writer, src='<en> Anyone who retains the ability to recognise beauty will never become old.', tgt='<de> Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt_lang_code='de')
                    evaluate(model, tokenizer, summary_writer, src='<en> Anyone who retains the ability to recognise beauty will never become old.', tgt='<fr> Quiconque conserve la capacité de reconnaître la beauté ne vieillira jamais.', tgt_lang_code='fr')
                    evaluate(model, tokenizer, summary_writer, src='<de> Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt='<vi> Người nào giữ được khả năng nhận biết cái đẹp sẽ không bao giờ già.', tgt_lang_code='vi')
                    evaluate(model, tokenizer, summary_writer, src='<vi> Người nào giữ được khả năng nhận biết cái đẹp sẽ không bao giờ già.', tgt='<en> Anyone who retains the ability to recognise beauty will never become old.', tgt_lang_code='en')

                    model.train()

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
                    predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs[:, :-1], src_key_padding_mask, tgt_key_padding_mask)
            else:
                predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs[:, :-1], src_key_padding_mask, tgt_key_padding_mask)

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

        # if rank == 0:
        if True:
            summary_writer.add_scalar('Validation Loss', losses.avg, steps)
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            viz_model(model, tokenizer, summary_writer, "<en> Anyone who retains the ability to recognise beauty will never become old.", "<de> Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

        return losses.avg

def evaluate(model, tokenizer, summary_writer, src, tgt, tgt_lang_code):
    global steps

    model.eval()
    best, _ = utils.beam_search_translate(args, src, model, tokenizer, tgt_lang_code, device=args.device, beam_size=4, length_norm_coefficient=0.6)

    debug_validate_table = PrettyTable([f"Test Source", f"Test Prediction", f"Test Target"])
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

    utils.init_transformer_weights(args, model, tie_embeddings=args.tie_embeddings)

    start_epoch = 0
    # if rank == 0:
    if True:
        if os.path.exists(os.path.join(run_dir, 'transformer_checkpoint.pth.tar')):
            start_epoch = utils.load_checkpoint(model, optimizer, os.path.join(run_dir, 'transformer_checkpoint.pth.tar'))
        utils.save_checkpoint(start_epoch, model, optimizer, f"runs/{args.run_name}/")

    # dist.barrier()

    start_epoch = utils.load_checkpoint(model, optimizer, os.path.join(run_dir, 'transformer_checkpoint.pth.tar'))

    model = model.to(rank)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if args.torch_compile_model:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = int(args.dynamo_cache_size_limit)

        # model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        model = torch.compile(model, dynamic=True)

    utils.print_model(model)

    return start_epoch, model, optimizer

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'  # You can use any free port

#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()

# def train(rank, world_size):
def train(rank):
    # setup(rank, world_size)

    summary_writer = SummaryWriter(log_dir=run_dir)

    tokenizer = Tokenizer.from_file("tokenizers/tokenizer_collated.json")

    start_epoch, model, optimizer = load_model(args, rank, tokenizer)

    criterion = LabelSmoothedCE(args, rank, eps=args.label_smoothing).to(args.device)

    if args.early_stop:
        early_stopping = utils.EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
    else:
        early_stopping = None

    # if rank == 0:
    if True:
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
            for n in tqdm(range(args.n_files - (world_size - 1)), desc=f"Epoch {epoch + 1}/{args.epochs}"):
                # train_loader = dataloader.get_generator(n, tokenizer, 'data', 'train', args.tokens_in_batch)(rank)
                # val_loader = dataloader.get_generator(0, tokenizer, 'data', 'validation', args.tokens_in_batch)(rank)

                data_idx = n + rank

                train_loader = dataloader.SequenceLoader(args, tokenizer, tokenizer, 'data', f"train_{data_idx}.src", f"train_{data_idx}.tgt", args.tokens_in_batch, for_training=True)
                train_loader.create_batches()

                train_epoch(rank, model, epoch, train_loader, scaler, criterion, optimizer, summary_writer, tokenizer, early_stopping)

                # if rank == 0:
                if True:
                    val_loader = dataloader.SequenceLoader(args, tokenizer, tokenizer, 'data', 'validation_0.src', 'validation_0.tgt', args.tokens_in_batch, for_training=False)
                    val_loader.create_batches()
                    val_loss_avg = validate_epoch(rank, model, epoch, val_loader, scaler, criterion, summary_writer, tokenizer)

                    utils.save_checkpoint(epoch, model, optimizer, prefix=run_dir)

                    if early_stopping is not None:
                        if early_stopping(val_loss_avg):
                            print("Early stopping")
                            utils.average_checkpoints(args.epochs, optimizer, run_dir, args.early_stop_num_latest_checkpoints_to_avg, model_name_prefix='step')
                            break

        time_taken = time.time() - start

        # if rank == 0:
        if True:
            print(f"Training complete. Averaging checkpoints...")
            utils.average_checkpoints(args.epochs, optimizer, run_dir, model_name_prefix='step')

            test_loader = dataloader.get_generator(0, tokenizer, 'data', 'validation', args.tokens_in_batch, rank)

            print("Training complete. Scoring with sacrebleu...")
            print(utils.sacrebleu_evaluate(args, run_dir, tokenizer, model, device=args.device, sacrebleu_in_python=True, test_loader=test_loader).score, time_taken, utils.count_parameters(model))
    finally:
        # cleanup()
        pass

if __name__ == "__main__":
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # world_size = torch.cuda.device_count()
    # mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

    train(args.device)
