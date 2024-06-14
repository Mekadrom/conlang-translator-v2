from criteria.labelsmooth import LabelSmoothedCE
from dataloader import SequenceLoader
from modules import transformer, utils
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import glob
import io
import matplotlib.pyplot as plt
import os
import seaborn as sns
import supreme_tokenizer
import time
import torch
import torch.nn as nn
import torch.optim as optim

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

args, unk = utils.get_args()

run_dir = os.path.join('runs', args.run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

summary_writer = SummaryWriter(log_dir=run_dir)

tokenizer = supreme_tokenizer.SupremeTokenizer(16384)

model = transformer.Transformer(args, tokenizer.total_vocab_size())
model = model.to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

if args.torch_compile_model:
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.cache_size_limit = int(args.dynamo_cache_size_limit)
    compiled_model = torch.compile(model)
else:
    compiled_model = model

criterion = LabelSmoothedCE(args, eps=args.label_smoothing).to(args.device)

if args.early_stop:
    early_stopping = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
else:
    early_stopping = None

utils.print_model(model)
print(f"Optimizer: {optimizer}")
print(f"Criterion: {criterion}")

if args.save_initial_checkpoint:
    utils.save_checkpoint(-1, model, optimizer, f"runs/{args.run_name}/")

sacrebleu_epochs = []

train_data_files = glob.glob(f"data/train_*")
val_data_files = glob.glob(f"data/validation_*")

if len(train_data_files) != len(val_data_files):
    raise ValueError("Number of train and validation files do not match.")

if len(train_data_files) != len(tokenizer.langs):
    raise ValueError("Number of train files and languages do not match.")

def load_data(tokens_in_batch, run_dir, tokenizer, pad_to_length=None):
    print('Loading training data SequenceLoader...')
    train_loader = SequenceLoader(
        tokenizer=tokenizer,
        data_files=train_data_files,
        tokens_in_batch=tokens_in_batch,
        for_training=True,
        pad_to_length=pad_to_length
    )

    print('Loading validation data SequenceLoader...')
    val_loader = SequenceLoader(
        tokenizer=tokenizer,
        data_files=val_data_files,
        tokens_in_batch=tokens_in_batch,
        pad_to_length=pad_to_length
    )

    return train_loader, val_loader

train_loader, val_loader = load_data(args.tokens_in_batch, run_dir, tokenizer, pad_to_length=args.maxlen)

class Trainer:
    def __init__(self, args, tokenizer, model, compiled_model, optimizer, criterion, train_loader, val_loader, device, run_dir, summary_writer, early_stopping=None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.compiled_model = compiled_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = run_dir
        self.summary_writer = summary_writer
        self.early_stopping = early_stopping

    def train(self, model_name_prefix=''):
        self.steps = 0
        self.start_epoch = self.args.start_epoch
        self.epochs = (self.args.n_steps // (self.train_loader.n_batches // self.args.batches_per_step)) + 1

        print(f"Training for {self.epochs} epochs...")
        start = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            self.steps = (epoch * self.train_loader.n_batches // self.args.batches_per_step)

            self.train_loader.create_batches()
            self.train_epoch(self.compiled_model, epoch=epoch)

            self.val_loader.create_batches()
            val_loss_avg = self.validate_epoch(self.model)

            utils.save_checkpoint(epoch, self.model, self.optimizer, prefix=f"{self.run_dir}/{model_name_prefix}")

            if self.early_stopping is not None:
                if self.early_stopping(val_loss_avg):
                    print("Early stopping")
                    utils.average_checkpoints(self.epochs, self.optimizer, self.run_dir, self.args.early_stop_num_latest_checkpoints_to_avg, model_name_prefix='step')

                    print(f"Training complete. Evaluating one last time...")
                    self.val_loader.create_batches()
                    self.validate_epoch(self.model)
                    break

        time_taken = time.time() - start

        # recalculate steps to make sure validation data is updated with correct steps
        self.steps = (self.epochs * self.train_loader.n_batches // self.args.batches_per_step)

        print(f"Training complete. Averaging checkpoints...")
        utils.average_checkpoints(self.epochs, self.optimizer, self.run_dir, model_name_prefix='step')

        print(f"Training complete. Evaluating one last time...")
        self.val_loader.create_batches()
        self.validate_epoch(self.model)

        print("Training complete. Scoring with sacrebleu...")
        print(utils.sacrebleu_evaluate(self.args, self.run_dir, self.tokenizer, self.model, device=self.device, sacrebleu_in_python=True, test_loader=self.val_loader).score, time_taken, utils.count_parameters(self.model))

        if self.args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                self.viz_model(0, self.model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

    def train_epoch(self, model, epoch):
        # training mode enables dropout
        model.train()

        data_time = utils.AverageMeter()
        step_time = utils.AverageMeter()
        total_losses = utils.AverageMeter()
        translation_losses = utils.AverageMeter()
        encoder_moe_gating_variance_losses = utils.AverageMeter()
        decoder_moe_gating_variance_losses = utils.AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths, src_langs, tgt_langs) in enumerate(self.train_loader):
            src_seqs = src_seqs.to(self.device) # (N, max_source_sequence_pad_length_this_batch)
            tgt_seqs = tgt_seqs.to(self.device) # (N, max_target_sequence_pad_length_this_batch)
            src_seq_lengths = src_seq_lengths.to(self.device) # (N)
            tgt_seq_lengths = tgt_seq_lengths.to(self.device) # (N)

            src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
            tgt_key_padding_mask = tgt_seqs == 0 # (N, max_target_sequence_pad_length_this_batch)

            data_time.update(time.time() - start_data_time)

            predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask) # (N, max_target_sequence_pad_length_this_batch, vocab_size)

            if self.args.moe_diversity_loss_coefficient > 0 and epoch >= self.args.moe_diversity_inclusion_epoch:
                encoder_moe_gating_variances = torch.stack(encoder_moe_gating_variances).std(dim=0).mean()
                decoder_moe_gating_variances = torch.stack(decoder_moe_gating_variances).std(dim=0).mean()
                moe_diversity_loss = (encoder_moe_gating_variances + decoder_moe_gating_variances) / 2
                encoder_moe_gating_variance_losses.update(encoder_moe_gating_variances.item(), 1)
                decoder_moe_gating_variance_losses.update(decoder_moe_gating_variances.item(), 1)

                moe_diversity_loss = moe_diversity_loss * self.args.moe_diversity_loss_coefficient
            else:
                moe_diversity_loss = 0

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            translation_loss = self.criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1) # scalar

            translation_losses.update(translation_loss.item(), (tgt_seq_lengths - 1).sum().item())

            loss = translation_loss + moe_diversity_loss

            (loss / self.args.batches_per_step).backward()

            total_losses.update(loss.item(), (tgt_seq_lengths - 1).sum().item())

            # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
            if (i + 1) % self.args.batches_per_step == 0:
                if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.steps += 1

                utils.change_lr(self.optimizer, new_lr=utils.get_lr(self.steps, self.args.d_model, self.args.warmup_steps))

                step_time.update(time.time() - start_step_time)

                if self.steps % self.args.print_frequency == 0:
                    print('Epoch {0}/{1}-----Batch {2}/{3}-----Step {4}/{5}-----Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                          'Loss {total_losses.val:.4f} ({total_losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(epoch + 1, self.epochs, i + 1,  self.train_loader.n_batches, self.steps, self.n_steps, step_time=step_time, data_time=data_time, total_losses=total_losses, early_stop_counter=self.early_stopping.counter if self.early_stopping is not None else 0, early_stop_patience=self.early_stopping.patience if self.early_stopping is not None else 0))
                    self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', src_lang='en', tgt_lang='de')

                self.summary_writer.add_scalar('Translation Training Loss', translation_losses.avg, self.steps)
                self.summary_writer.add_scalar('Training Loss', total_losses.avg, self.steps)
                if moe_diversity_loss > 0:
                    self.summary_writer.add_scalar('Encoder MoE Gating Variances', encoder_moe_gating_variance_losses.avg, self.steps)
                    self.summary_writer.add_scalar('Decoder MoE Gating Variances', decoder_moe_gating_variance_losses.avg, self.steps)

                start_step_time = time.time()

                # 'epoch' is 0-indexed
                # early stopping requires the ability to average the last few checkpoints so just save all of them
                if (epoch in [self.epochs - 1, self.epochs - 2] or self.args.early_stop) and self.steps % 1500 == 0:
                    utils.save_checkpoint(epoch, self.model, self.optimizer, prefix=f"{self.run_dir}/step{str(self.steps)}_")
            start_data_time = time.time()
    
    def validate_epoch(self, model):
        model.eval()

        with torch.no_grad():
            losses = utils.AverageMeter()
            for (src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths, src_langs, tgt_langs) in tqdm(self.val_loader, total=self.val_loader.n_batches):
                src_seqs = src_seqs.to(self.device) # (1, max_source_sequence_pad_length_this_batch)
                tgt_seqs = tgt_seqs.to(self.device) # (1, max_target_sequence_pad_length_this_batch)
                src_seq_lengths = src_seq_lengths.to(self.device) # (1)
                tgt_seq_lengths = tgt_seq_lengths.to(self.device) # (1)

                src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
                tgt_key_padding_mask = tgt_seqs == 0 # (N, max_target_sequence_pad_length_this_batch)

                predicted_sequences = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)[0] # (N, max_target_sequence_pad_length_this_batch, vocab_size)

                # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
                # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
                # Therefore, pads start after (length - 1) positions
                loss = self.criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1) # scalar

                losses.update(loss.item(), (tgt_seq_lengths - 1).sum().item())

            self.summary_writer.add_scalar('Validation Loss', losses.avg, self.steps)
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            self.viz_model(self.steps, model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", "en", "de")

            return losses.avg

    def evaluate(self, src, tgt, src_lang, tgt_lang):
        best, _ = utils.beam_search_translate(self.args, src, self.model, self.tokenizer, src_lang, tgt_lang, device=self.device, beam_size=4, length_norm_coefficient=0.6)

        debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
        debug_validate_table.add_row([src, best, tgt])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(debug_validate_table)

    def viz_attn_weights(self, stack_name, layer_num, n_head, activation_weights, attendee_tokens, attending_tokens):
        fig, ax = plt.subplots(figsize=(10, 10))
        s = sns.heatmap(activation_weights, square=True, annot=True, annot_kws={"fontsize":6}, fmt=".4f", xticklabels=attendee_tokens, yticklabels=attending_tokens, ax=ax)
        s.set(xlabel="Attending Tokens", ylabel="Attended Tokens", title=f"{stack_name}-Attn Layer {layer_num} Head {n_head} Weights")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return buf

    def viz_model(self, step, model, src, tgt, src_lang, tgt_lang):
        with torch.no_grad():
            model.eval()

            input_sequence = torch.LongTensor(self.tokenizer.encode(src, src_lang, eos=False)).unsqueeze(0).to(self.device) # (1, input_sequence_length)
            input_tokens = [self.tokenizer.decode([id.item()], src_lang)[0] for id in input_sequence.squeeze(0)]
            input_sequence_length = input_sequence.size(1)

            # pad input sequence to args.maxlen
            if self.args.use_infinite_attention or True:
                input_sequence = torch.cat([input_sequence, torch.zeros([1, self.args.maxlen - input_sequence.size(1)], dtype=torch.long, device=input_sequence.device)], dim=1)

            target_sequence = torch.LongTensor(self.tokenizer.encode(tgt, tgt_lang, eos=True)).unsqueeze(0).to(self.device) # (1, target_sequence_length)
            target_tokens = [self.tokenizer.decode([id.item()], tgt_lang)[0] for id in target_sequence.squeeze(0)]
            target_sequence_length = target_sequence.size(1)

            # pad target sequence to args.maxlen
            if self.args.use_infinite_attention or True:
                target_sequence = torch.cat([target_sequence, torch.zeros([1, self.args.maxlen - target_sequence.size(1)], dtype=torch.long, device=target_sequence.device)], dim=1)

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
                        image_data = self.viz_attn_weights('Encoder-Self', e, i, attention_weight_grid[:, i, :input_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).cpu().detach().numpy(), input_tokens, input_tokens)
                        self.summary_writer.add_image(f"Encoder Layer {e} Head {i} Self-Attn Weights for Segment {a}", plt.imread(image_data), global_step=step, dataformats='HWC')

                input_sequence, _ = encoder_layer.fcn(sequences=input_sequence) # (N, pad_length, d_model)

            input_sequence = model.encoder.norm(input_sequence)

            target_sequence, _, _ = model.decoder.apply_embedding_transformation(target_sequence) # (N, pad_length, d_model)
            target_sequence = model.decoder.apply_positional_embedding(target_sequence) # (N, pad_length, d_model)

            for d, decoder_layer in enumerate(model.decoder.decoder_layers):
                target_sequence, attention_weights = decoder_layer.self_attn(target_sequence, target_sequence, target_sequence, tgt_key_padding_mask) # (N, pad_length, d_model)
                
                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                    # shape of attention_weight_grid will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Decoder-Self', d, i, attention_weight_grid[:, i, :target_sequence_length, :target_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, target_tokens)
                        self.summary_writer.add_image(f"Decoder Layer {d} Head {i} Self-Attn Weights for Segment {a}", plt.imread(image_data), global_step=step, dataformats='HWC')

                target_sequence, attention_weights = decoder_layer.cross_attn(target_sequence, input_sequence, input_sequence, src_key_padding_mask) # (N, pad_length, d_model)

                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                    # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Decoder-Cross', d, i, attention_weight_grid[:, i, :target_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, input_tokens)
                        self.summary_writer.add_image(f"Decoder Layer {d} Head {i} Cross-Attn Weights for Segment {a}", plt.imread(image_data), global_step=step, dataformats='HWC')

                target_sequence, _ = decoder_layer.fcn(target_sequence) # (N, pad_length, d_model)
