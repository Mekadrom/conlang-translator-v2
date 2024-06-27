from collections import OrderedDict
from datasets import load_dataset
from dataloader import SequenceLoader
from positional_encodings.torch_encodings import PositionalEncoding2D
from rotary_embedding_torch import RotaryEmbedding
from modules.swiglu import SwiGLU
from tqdm import tqdm

import argparse
import codecs
import glob
import math
import modules.transformer as transformer
import os
import sacrebleu
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import traceback
import yaml
import youtokentome

VOCAB_SIZES = {
    # 'afr': 8192,
    # 'amh': 3072,
    'cs': 3072,
    'de': 8192,
    'en': 16384,
    'et': 2048,
    'fi': 4096,
    'fr': 6144,
    # 'fuv': 3072,
    'gu': 1024,
    # 'hau': 4096,
    'hi': 1024,
    # 'ibo': 3072,
    # 'ja': 12288,
    # 'kam': 2048,
    # 'kin': 4096,
    'kk': 1024,
    # 'lin': 4096,
    'lt': 3072,
    # 'lug': 2048,
    # 'luo': 4096,
    'lv': 1024,
    # 'nso': 4096,
    # 'nya': 4096,
    # 'orm': 4096,
    'ro': 1024,
    'ru': 6144,
    # 'sna': 4096,
    # 'som': 2048,
    # 'ssw': 2048,
    # 'swh': 6144,
    'tr': 1024,
    # 'tsn': 4096,
    # 'tso': 2048,
    # 'umb': 2048,
    'vi': 7168,
    # 'wol': 3072,
    # 'xho': 10240,
    # 'yor': 10240,
    # 'zh': 15360,
    # 'zul': 2048,
    'con': 8192
}

# first n tokens are reserved for the <lang> tags at the beginning of src and tgt sequences
# from then on, each language's tokenizer handles the special tokens (pad, unk, bos, eos) on their own
TOTAL_VOCAB_SIZE = len(VOCAB_SIZES) + sum(VOCAB_SIZES.values())

def get_language_tokenizer_offset(lang):
    if lang.startswith("<") and lang.endswith(">"):
        lang = lang[1:-1]

    offset = len(VOCAB_SIZES)
    for l, vocab_size in VOCAB_SIZES.items():
        if l == lang:
            return offset
        offset += vocab_size
    raise ValueError(f"Language {lang[:min(20, len(lang))]} not found in VOCAB_SIZES")

def get_language_indicator_index(lang):
    return list(VOCAB_SIZES.keys()).index(lang)

def get_language_indicator_from_index(index):
    return list(VOCAB_SIZES.keys())[index]

def get_structured_data_paths(data_files):
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
    src_tgt_pairs = {}
    for data_file in data_files:
        file_name, single = data_file.split(".")
        split, pair = file_name.split("_")
        src, tgt = pair.split("-")

        if pair not in src_tgt_pairs.keys():
            src_tgt_pairs[pair] = {'src': None, 'tgt': None }

        src_tgt_pairs[pair]['src' if src == single else 'tgt'] = data_file

    return src_tgt_pairs

def init_transformer_weights(args, model, tie_embeddings=True):
    # Glorot uniform initialization with a gain of self.args.init_weights_gain
    for p in model.parameters():
        # Glorot initialization needs at least two dimensions on the tensor
        if p.dim() > 1:
            if args.init_weights_from in ['glorot_uniform', 'xavier_uniform']:
                nn.init.xavier_uniform_(p, gain=args.init_weights_gain)
            elif args.init_weights_from in ['glorot_normal', 'xavier_normal']:
                nn.init.xavier_normal_(p, gain=args.init_weights_gain)
            elif args.init_weights_from == 'kaiming_uniform':
                nn.init.kaiming_uniform_(p)
            elif args.init_weights_from == 'kaiming_normal':
                nn.init.kaiming_normal_(p)
            elif args.init_weights_from == 'orthogonal':
                nn.init.orthogonal_(p)
            else:
                raise Exception(f"Unknown weight initialization method: {args.init_weights_from}")

    # Share weights between the embedding layers and the logit layer

    if isinstance(model, transformer.Transformer):
        nn.init.normal_(model.encoder.embedding.weight, mean=0., std=args.d_model**-0.5)
        model.decoder.embedding.weight = model.encoder.embedding.weight

        if tie_embeddings:
            model.decoder.classifier.weight = model.decoder.embedding.weight
    elif isinstance(model, transformer.Decoder):
        if tie_embeddings:
            model.classifier.weight = model.embedding.weight

    print("Model initialized.")

def get_lr(step, d_model, warmup_steps):
    """
    The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repository.

    :param step: training step number
    :param d_model: size of vectors throughout the transformer model
    :param warmup_steps: number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official T2T repo
    :return: updated learning rate
    """
    return 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))

def get_buffered_positional_encoding(args, d_model, maxlen=100, num_dims=1):
    """
    Computes positional encoding as defined in the paper.

    :param d_model: size of vectors throughout the transformer model
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """
    if num_dims == 1:
        positional_encoding = torch.zeros((maxlen, d_model)) # (max_length, d_model)
        for i in range(maxlen):
            for k in range(d_model):
                if k % 2 == 0:
                    positional_encoding[i, k] = math.sin(i / math.pow(10000, k / d_model))
                else:
                    positional_encoding[i, k] = math.cos(i / math.pow(10000, (k - 1) / d_model))
        positional_encoding = positional_encoding.unsqueeze(0) # (1, max_length, d_model)
    elif num_dims == 2:
        positional_encoding_2d = PositionalEncoding2D(args.positional_encoding_dim).to(args.device)
        positional_encoding = torch.zeros((1, maxlen, maxlen, args.positional_encoding_dim))
        positional_encoding = positional_encoding_2d(positional_encoding.to(args.device))
    return positional_encoding  # (1, max_length, d_model) or (1, max_length, max_length, d_model)

def get_positional_encoding(args):
    if args.positional_encoding_type == 'sinusoidal' or args.positional_encoding_type == 'buffer':
        positional_encoding = get_buffered_positional_encoding(
            args,
            d_model=args.d_model,
            maxlen=args.maxlen+1,
        ).to(args.device)
        positional_encoding.requires_grad = bool(args.learnable_positional_encoding)
    elif args.positional_encoding_type == 'rotary':
        positional_encoding = RotaryEmbedding(dim=args.positional_encoding_dim)
    return positional_encoding

def load_data(args, tokens_in_batch, tokenizer, pad_to_length=None):
    print('Loading training data SequenceLoader...')
    train_loader = SequenceLoader(
        args,
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        data_folder=os.path.join('data'),
        source_suffix="src",
        target_suffix='tgt',
        split=f"train",
        tokens_in_batch=tokens_in_batch,
        pad_to_length=pad_to_length
    )

    print('Loading validation data SequenceLoader...')
    val_loader = SequenceLoader(
        args,
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        data_folder=os.path.join('data'),
        source_suffix="src",
        target_suffix='tgt',
        split=f"validation",
        tokens_in_batch=tokens_in_batch,
        pad_to_length=pad_to_length
    )

    return train_loader, val_loader

def print_model(model):
    print(f"Model structure: \n {model}")
    print(f'The model has {count_parameters(model):,} total parameters')
    print(f'The model has {sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad):,} non-zero total parameters')

    def tensor_in(tensor, tensor_list):
        for t in tensor_list:
            if tensor is t:
                return True
        return False

    already_counted = []
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad and not tensor_in(param, already_counted):
            # print(f"Layer {name} has {param.numel():,} parameters and {torch.count_nonzero(param).item():,} non-zero parameters")
            total_params += param.numel()
            already_counted.append(param)

    print(f'The model has {total_params:,} trainable parameters')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(epoch, model, optimizer, prefix=''):
    """
    Checkpoint saver. Each save overwrites previous save.

    :param epoch: epoch number (0-indexed)
    :param model: transformer model
    :param optimizer: optimized
    :param prefix: checkpoint filename prefix
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    filename = prefix + 'transformer_checkpoint.pth.tar'
    torch.save(state, filename)

def load_checkpoint(model, optimizer, path):
    state = torch.load(path)
    model.load_state_dict(state['model'].state_dict())
    optimizer.load_state_dict(state['optimizer'].state_dict())
    return state['epoch']

def change_lr(optimizer, new_lr):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be changed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

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

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clean_decoded_text(decoded_text):
    tokens = decoded_text.split()
    cleaned_tokens = []
    for i, token in enumerate(tokens):
        if i == 0:
            cleaned_tokens.append(token.lstrip('▁'))
        else:
            cleaned_tokens.append(token.replace('▁', ' '))
    return ''.join(cleaned_tokens)

def beam_search_translate(args, src, model, tokenizer, tgt_lang_code, device, beam_size=4, length_norm_coefficient=0.6):
    """
    Translates a source language sequence to the target language, with beam search decoding.

    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size
    :param length_norm_coefficient: co-efficient for normalizing decoded sequences' scores by their lengths
    :return: the best hypothesis, and all candidate hypotheses
    """
    with torch.no_grad():
        # Beam size
        k = beam_size

        # Minimum number of hypotheses to complete
        n_completed_hypotheses = min(k, 10)

        # If the source sequence is a string, convert to a tensor of IDs
        if isinstance(src, str):
            encoder_sequences = tokenizer.encode(src).ids
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0) # (1, source_sequence_length)
        else:
            encoder_sequences = src

        encoder_sequences = encoder_sequences.to(device) # (1, source_sequence_length)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(device) # (1)

        src_key_padding_mask = (encoder_sequences == 0).to(device) # (1, source_sequence_length)
        
        # Encode
        encoder_sequences, gating_variances = model.encoder(encoder_sequences, src_key_padding_mask) # (1, source_sequence_length, d_model)

        # Our hypothesis to begin with is just <bos>
        hypotheses = torch.LongTensor([[2, tokenizer.encode(f'<{tgt_lang_code}>').ids[0]]]).to(device) # (1, 1) (bos == 2)

        # Tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(device) # (1)

        # Lists to store completed hypotheses and their scores
        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        # Start decoding
        step = 1

        # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<bos>"
        while True:
            s = hypotheses.size(0)

            tgt_key_padding_masks = torch.zeros(s, hypotheses.size(1)).to(device).bool()

            decoder_sequences, gating_variances = model.decoder(
                hypotheses,
                encoder_sequences.repeat(s, 1, 1),
                src_key_padding_mask.repeat(s, 1), # (s, 1)
                tgt_key_padding_masks
            )

            # Scores at this step
            scores = decoder_sequences[:, -1, :] # (s, tgt_vocab_size)
            scores = F.log_softmax(scores, dim=-1) # (s, tgt_vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores # (s, tgt_vocab_size)

            # Unroll and find top k scores, and their unrolled indices
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True) # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // tokenizer.get_vocab_size() # (k)
            next_word_indices = unrolled_indices % tokenizer.get_vocab_size() # (k)

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1) # (k, step + 1)

            # Which of these new hypotheses are complete (reached <EOS>)?
            complete = next_word_indices == 3 # (k), bool (EOS == 3)

            # Set aside completed hypotheses and their scores normalized by their lengths
            # For the length normalization formula, see
            # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            # Stop if we have completed enough hypotheses
            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            # Else, continue with incomplete hypotheses
            hypotheses = top_k_hypotheses[~complete] # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete] # (s)

            # Stop if things have been going on for too long
            if step > args.maxlen:
                break
            step += 1

        # If there is not a single completed hypothesis, use partial hypotheses
        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()

        # Decode the hypotheses
        all_hypotheses = list()
        decoded = [clean_decoded_text(tokenizer.decode(completed_hypothesis, skip_special_tokens=True)) for completed_hypothesis in completed_hypotheses]
        for i, h in enumerate(decoded):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        # Find the best scoring completed hypothesis
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses

def average_checkpoints(epoch, optimizer, source_folder, num_latest_checkpoints=None, model_name_prefix='step', model_name_suffix='_transformer_checkpoint.pth.tar'):
    # Get list of checkpoint names
    checkpoint_names = [f for f in os.listdir(source_folder) if f.startswith(model_name_prefix) and f.endswith(model_name_suffix)]
    assert len(checkpoint_names) > 0, "Did not find any checkpoints!"

    # order the checkpoint names by step number
    checkpoint_names = sorted(checkpoint_names, key=lambda x: int(x[len(model_name_prefix):-len(model_name_suffix)]))

    if num_latest_checkpoints is not None:
        # only take X latest checkpoints
        checkpoint_names = checkpoint_names[-num_latest_checkpoints:]

    # Average parameters from checkpoints
    averaged_params = OrderedDict()
    for c in tqdm(checkpoint_names, desc="Averaging checkpoints"):
        checkpoint = torch.load(os.path.join(source_folder, c))['model']
        checkpoint_params = checkpoint.state_dict()
        checkpoint_param_names = checkpoint_params.keys()
        for param_name in checkpoint_param_names:
            if param_name not in averaged_params:
                averaged_params[param_name] = checkpoint_params[param_name].clone() * 1 / len(checkpoint_names)
            else:
                averaged_params[param_name] += checkpoint_params[param_name] * 1 / len(checkpoint_names)

    # Use one of the checkpoints as a surrogate to load the averaged parameters into
    averaged_checkpoint = torch.load(os.path.join(source_folder, checkpoint_names[0]))['model']
    for param_name in averaged_checkpoint.state_dict().keys():
        assert param_name in averaged_params
    averaged_checkpoint.load_state_dict(averaged_params)

    # Save averaged checkpoint
    torch.save({'epoch': epoch, 'model': averaged_checkpoint, 'optim': optimizer}, f"{source_folder}/averaged_transformer_checkpoint.pth.tar")

def sacrebleu_evaluate(args, run_dir, tokenizer, model, device, sacrebleu_in_python, test_loader=None):
    """
    Returns None when command line sacrebleu is used
    """

    before_nanos = time.time_ns()

    bleu_score = None

    # Evaluate
    with torch.no_grad():
        hypotheses = list()
        references = list()
        for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(tqdm(test_loader, total=test_loader.n_batches)):
            hypotheses.append(beam_search_translate(args, src=source_sequence, tokenizer=tokenizer, device=device, model=model, beam_size=4, length_norm_coefficient=0.6)[0])
            references.extend(tokenizer.decode_all(target_sequence.tolist(), ignore_ids=[0, 2, 3]))

        if sacrebleu_in_python:
            print("\n13a tokenization, cased:\n")
            bleu_score = sacrebleu.corpus_bleu(hypotheses, [references])
            print(bleu_score)
            print("\n13a tokenization, caseless:\n")
            print(sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True))
            print("\nInternational tokenization, cased:\n")
            print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'))
            print("\nInternational tokenization, caseless:\n")
            print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True))
            print("\n")
        else:
            cat_command = "cat" if os.name == "posix" else "type"

            with codecs.open(os.path.join(run_dir, "translated_test.tgt"), "w", encoding="utf-8") as f:
                f.write("\n".join(hypotheses))

            print("\n13a tokenization, cased:\n")
            os.system(f"{cat_command} translated_test.tgt | sacrebleu -t wmt14/full -l en-de")
            print("\n13a tokenization, caseless:\n")
            os.system(f"{cat_command} translated_test.tgt | sacrebleu -t wmt14/full -l en-de -lc")
            print("\nInternational tokenization, cased:\n")
            os.system(f"{cat_command} translated_test.tgt | sacrebleu -t wmt14/full -l en-de -tok intl")
            print("\nInternational tokenization, caseless:\n")
            os.system(f"{cat_command} translated_test.tgt | sacrebleu -t wmt14/full -l en-de -tok intl -lc")
            print("\n")
        print(
            "The first value (13a tokenization, cased) is how the BLEU score is officially calculated by WMT (mteval-v13a.pl). \nThis is probably not how it is calculated in the 'Attention Is All You Need' paper, however.\nSee https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-380970191 for more details.\n")
        
    after_nanos = time.time_ns()

    print(f"Time taken for sacrebleu evaluation: {(after_nanos - before_nanos) / 1e9} seconds")

    return bleu_score

def create_activation_function(d_in, activation_function_name):
    if activation_function_name == 'relu':
        return nn.ReLU()
    elif activation_function_name == 'gelu':
        return nn.GELU()
    elif activation_function_name == 'elu':
        return nn.ELU()
    elif activation_function_name == 'selu':
        return nn.SELU()
    elif activation_function_name == 'prelu':
        return nn.PReLU()
    elif activation_function_name == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_function_name == 'swiglu':
        return SwiGLU(d_in)
    elif activation_function_name == 'none':
        return nn.Identity()
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")

class YamlDict(dict):
    def __init__(self, *args, **kwargs):
        super(YamlDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, name):
        return self.__getitem__(name) if name in self else super().__getattribute__(name)

def load_yaml(file_path, ovr_args):
    file_path_dir = os.path.dirname(file_path)
    print(f"loading configs from {file_path_dir}")
    with open(os.path.join(file_path_dir, 'default.yaml'), 'r') as default_config:
        with open(file_path, 'r') as f:
            y = yaml.safe_load(default_config)
            y.update(yaml.safe_load(f))
            if ovr_args is not None:
                y.update(ovr_args)
            return YamlDict(y)

def get_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--run_name', type=str, required=True)
    argparser.add_argument('--config_file_path', type=str, required=True)

    argsparser_args, unk = argparser.parse_known_args()

    # convert unk list to dict
    unk = {unk[i][2:]: unk[i + 1] for i in range(0, len(unk), 2)}

    if len(unk) > 0:
        print(f"unknown arguments: {unk}")

    args = load_yaml(argsparser_args.config_file_path, unk)
    args.__setattr__('run_name', argsparser_args.run_name)

    print(f"args: {args}")

    if args.n_gqa_groups == 0 or args.n_heads == 0:
        print("it is not recommended to not have any multi-head attention layers")
        exit(1)

    if hasattr(args, 'tokens_in_batch'):
        args.__setattr__('batches_per_step', args.target_tokens_per_batch // args.tokens_in_batch)
    args.__setattr__('lr', get_lr(step=1, d_model=args.d_model, warmup_steps=args.warmup_steps))

    torch.set_printoptions(profile='full')

    torch.autograd.set_detect_anomaly(args.detect_nans)
    cudnn.benchmark = bool(args.cudnn_benchmark)

    return args, unk
