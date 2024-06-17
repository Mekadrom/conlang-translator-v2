import admin_torch
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x, y):
        return x + y

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x):

        x_sqr = x**2
        RMS = torch.rsqrt(x_sqr.mean(dim = -1, keepdim = True) + self.eps)
        new_x = x * RMS
        new_x = new_x * self.weight

        return new_x

class MultiHeadAttention(nn.Module):
    def __init__(self, args, self_attn, in_decoder=False, norm=nn.LayerNorm):
        super(MultiHeadAttention, self).__init__()

        self.args = args
        self.self_attn = self_attn
        self.in_decoder = in_decoder

        if args.positional_encoding_type == 'rotary':
            self.rotary_embedding = utils.get_positional_encoding(args)

        self.n_q_heads = args.n_gqa_groups * args.n_heads
        self.n_heads = args.n_heads
        self.n_gqa_groups = args.n_gqa_groups

        self.d_queries = args.d_queries
        self.d_keys = args.d_queries
        self.d_values = args.d_values

        # A linear projection to cast (n_kv_heads sets of) queries from the input query sequences
        self.cast_queries = nn.Linear(args.d_model, self.n_q_heads * self.d_queries) # (N, query_sequence_pad_length, n_kv_heads * d_queries)
        # A linear projection to cast (n_kv_heads sets of) keys and values from the input reference sequences
        self.cast_keys = nn.Linear(args.d_model, args.n_heads * self.d_keys) # (N, key_value_sequence_pad_length, n_kv_heads * d_keys)
        self.cast_values = nn.Linear(args.d_model, args.n_heads * self.d_values) # (N, key_value_sequence_pad_length, n_kv_heads * d_values)

        # a linear projection to cast (n_q_heads sets of) computed attention-weighted vectors to output vectors
        self.mha_cast_output = nn.Linear(args.n_heads * self.d_values, args.d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.norm = norm(args.d_model, args.norm_eps)

        self.dropout = nn.Dropout(args.dropout)

        if 'heads_activation' in args:
            self.heads_activation = utils.create_activation_function(args.d_model, args.heads_activation)
        else:
            self.heads_activation = None

        if args.use_infinite_attention:
            assert args.maxlen % args.infinite_attention_n_segments == 0, "maxlen must be divisible by infinite_attention_n_segments"

            self.beta = nn.Parameter(torch.ones((1,)))
            self.elu = nn.ELU()
            self.register_buffer('causal_mask', torch.tril(torch.ones((args.maxlen // args.infinite_attention_n_segments) + 1, (args.maxlen // args.infinite_attention_n_segments) + 1).to(args.device)))
        else:
            self.beta = None
            self.elu = None
            self.register_buffer('causal_mask', torch.tril(torch.ones(args.maxlen + 1, args.maxlen + 1).to(args.device)))

    def mask_attention(self, attention_weights, key_padding_mask):
        # mask away tokens by setting such weights to a large negative number, so that they evaluate to 0 under the softmax

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == attention_weights.shape[0], f"batch dimension for padding is wrong: {key_padding_mask.shape[0]} != {attention_weights.shape[0]}. overall shape: {key_padding_mask.shape} != {attention_weights.shape}"
            assert key_padding_mask.shape[1] == attention_weights.shape[3], f"padding mask length is wrong: {key_padding_mask.shape[1]} != {attention_weights.shape[3]}. overall shape: {key_padding_mask.shape} != {attention_weights.shape}"

            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

            attention_weights = attention_weights.masked_fill_(key_padding_mask, -float('inf'))

        if self.self_attn:
            attention_weights = attention_weights.masked_fill_(self.causal_mask[:attention_weights.shape[-2], :attention_weights.shape[-1]] == 0, -float('inf'))

        return attention_weights

    def forward(self, query_sequences, key_sequences, value_sequences, key_padding_mask=None):
        query_sequences = self.norm(query_sequences)

        # if this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self.self_attn:
            key_sequences = self.norm(key_sequences)
            value_sequences = self.norm(value_sequences)

        q_heads = self.cast_queries(query_sequences)
        k_heads = self.cast_keys(key_sequences)
        v_heads = self.cast_values(value_sequences)

        if self.heads_activation is not None:
            q_heads = self.heads_activation(q_heads)
            k_heads = self.heads_activation(k_heads)
            v_heads = self.heads_activation(v_heads)

        N = q_heads.size(0) # batch size (N) in number of sequences
        t = q_heads.size(1) # query sequence padded lengths
        T = k_heads.size(1) # key-value sequence padded lengths

        # Split the last dimension by the n_kv_heads subspaces
        q_heads = q_heads.contiguous().view(N, t, self.n_gqa_groups, self.n_heads, self.d_queries) # (N, query_sequence_pad_length, n_gqa_groups, n_heads, d_queries)
        k_heads = k_heads.contiguous().view(N, T, self.n_heads, self.d_keys) # (N, key_value_sequence_pad_length, n_heads, d_keys)
        v_heads = v_heads.contiguous().view(N, T, self.n_heads, self.d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        q_heads = q_heads.permute(0, 2, 3, 1, 4) # Nghtd
        k_heads = k_heads.permute(0, 2, 1, 3) # NhTd
        v_heads = v_heads.permute(0, 2, 1, 3) # NhTv

        if hasattr(self, 'rotary_embedding') and self.rotary_embedding is not None:
            q_heads = self.rotary_embedding.rotate_queries_or_keys(q_heads, seq_dim=-2)
            k_heads = self.rotary_embedding.rotate_queries_or_keys(k_heads.unsqueeze(0), seq_dim=-2).squeeze(0) # adds a singleton dimension for the rotation operation and then removes it for the torch compiler

        attention_weights_for_visualization = []
        if self.args.use_infinite_attention:
            # infinite attention
            memory = torch.zeros((self.n_head, self.d_queries, self.d_queries)).to(self.device)
            z = torch.zeros((self.n_head, self.d_queries, 1)).to(self.device)

            q_heads = q_heads.view(N, self.n_gqa_groups, self.n_heads, self.args.infinite_attention_n_segments, t // self.args.infinite_attention_n_segments, self.d_queries) # Nghitq
            k_heads = k_heads.view(N, self.n_heads, self.args.infinite_attention_n_segments, T // self.args.infinite_attention_n_segments, self.d_keys) # NhiTq
            v_heads = v_heads.view(N, self.n_heads, self.args.infinite_attention_n_segments, T // self.args.infinite_attention_n_segments, self.d_values) # NhiTv

            output = []
            for idx in range(self.args.infinite_attention_n_segments):
                sigma_q = self.elu(q_heads[:, :, :, idx, :, :]) + 1.0
                sigma_k = self.elu(k_heads[:, :, idx, :, :]) + 1.0

                A_mem = (sigma_q @ memory) / ((sigma_q @ z) + (1e-6))

                attention_weights = q_heads[:, :, idx, :, :] @ k_heads[:, :, idx, :, :].transpose(-2, -1)

                # scaled attention
                attention_weights = (1.0 / (self.d_queries ** 0.5)) * attention_weights
                # attention_weights = 30.0 * torch.tanh(attention_weights / 30.0) # grok version of scaled attention

                attention_weights = self.mask_attention(attention_weights, key_padding_mask)

                attention_weights = self.softmax(attention_weights)

                attention_weights_for_visualization.append(attention_weights.clone().detach().contiguous().view(N, self.n_gqa_groups, self.n_heads, t // self.args.infinite_attention_n_segments, T // self.args.infinite_attention_n_segments))

                # not included in paper for some reason? experiment
                # attention_weights = self.dropout(attention_weights)
                attention_weights = attention_weights @ v_heads[:, :, idx, :, :]

                attention_weights = (F.sigmoid(self.beta) * A_mem) + ((1 - F.sigmoid(self.beta)) * attention_weights)

                if self.infinite_attention_update == 'linear':
                    memory = memory + (sigma_k.transpose(-2, -1) @ v_heads[:, :, idx, :, :])
                else:
                    delta = (sigma_k @ memory) / ((sigma_k @ z) + 1e-6)
                    memory = memory + (sigma_k.transpose(-2, -1) @ (v_heads[:, :, idx, :, :] - delta))

                z = z + sigma_k.sum(dim=-2, keepdim=True)

                output.append(attention_weights)

            sequences = torch.concat(output, dim = 2) # NhiTv
        else:
            # regular attention
            # generate attention weights by taking the dot product of queries and keys
            attention_weights = torch.einsum('...ghtq,...hTq->...htT', q_heads, k_heads)

            # scaled attention
            attention_weights = (1.0 / (self.d_queries ** 0.5)) * attention_weights
            attention_weights = 30.0 * torch.tanh(attention_weights / 30.0) # grok version of scaled attention

            attention_weights = self.mask_attention(attention_weights, key_padding_mask)

            attention_weights = self.softmax(attention_weights)

            # for visualization, switch the kv_heads and q_per_kv_heads dimensions
            attention_weights_for_visualization.append(attention_weights.clone().detach())

            attention_weights = self.dropout(attention_weights)

            # Calculate sequences as the weighted sums of values based on these softmax weights
            sequences = torch.einsum('...htT,...hTv->...htv', attention_weights, v_heads)

            sequences = sequences.permute(0, 2, 1, 3)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(N, t, -1)

        sequences = self.dropout(sequences)

        sequences = self.mha_cast_output(sequences)

        return sequences, attention_weights_for_visualization

class SparseMoE(nn.Module):
    def __init__(self, args):
        super(SparseMoE, self).__init__()

        self.args = args

        self.expert_weights = nn.ModuleList([nn.Linear(args.d_model, args.d_inner) for _ in range(args.n_experts)])
        self.gating = nn.Linear(args.d_model, args.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        
        self.reset_parameters()

    def reset_parameters(self):
        for expert in self.expert_weights:
            nn.init.xavier_uniform_(expert.weight)
            nn.init.zeros_(expert.bias)

    def forward(self, sequences):
        N, P, D = sequences.shape

        # merge batch and sequence dimensions
        flat_sequences = sequences.view(-1, D) # (N * pad_length, d_model)
        gating_scores = self.softmax(self.gating(flat_sequences))

        top_k_indices = torch.topk(gating_scores, self.args.moe_top_k, dim=1).indices

        output = torch.zeros(N*P, self.expert_weights[0].out_features, device=sequences.device)

        for i in range(len(self.expert_weights)):
            expert_mask = top_k_indices == i
            expert_input = flat_sequences[expert_mask.any(dim=1)]
            expert_output = self.expert_weights[i](expert_input)

            output[expert_mask.any(dim=1)] += expert_output

        # record export choices to self.gating_variances for loss calculation to encourage diversity
        if self.training:
            gating_variances = torch.var(gating_scores, dim=0)
        else:
            gating_variances = None

        # normalize
        output /= self.args.moe_top_k

        return output.view(N, P, -1), gating_variances
    
class PositionWiseFCNetwork(nn.Module):
    def __init__(self, args, norm=nn.LayerNorm):
        super(PositionWiseFCNetwork, self).__init__()

        self.args = args

        self.layer_norm = norm(args.d_model, args.norm_eps)
        self.activation = utils.create_activation_function(args.d_inner, args.activation_function)
        self.dropout = nn.Dropout(args.dropout)
        
        if args.use_moe:
            self.expand = SparseMoE(args)
        else:
            self.expand = nn.Linear(args.d_model, args.d_inner)

        self.condense = nn.Linear(args.d_inner, args.d_model)

    def forward(self, sequences):
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        if type(self.expand) == nn.Linear:
            sequences = self.expand(sequences) # (N, pad_length, d_inner)
            gating_variances = None
        else:
            sequences, gating_variances = self.expand(sequences)

        sequences = self.activation(sequences)
        sequences = self.dropout(sequences)  # (N, pad_length, d_inner)

        sequences = self.condense(sequences)  # (N, pad_length, d_model)

        sequences = self.dropout(sequences) # (N, pad_length, d_model)

        return sequences, gating_variances

class EncoderLayer(nn.Module):
    def __init__(self, args, norm=nn.LayerNorm):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(args, self_attn=True, in_decoder=False, norm=norm)

        if args.use_admin:
            self.self_attn_residual = admin_torch.as_module(args.n_encoder_layers)
            self.fcn_residual = admin_torch.as_module(args.n_encoder_layers)
        else:
            self.self_attn_residual = Sum()
            self.fcn_residual = Sum()

        self.fcn = PositionWiseFCNetwork(args, norm=norm)

    def forward(self, encoder_sequences, key_padding_mask):
        self_attn, _ = self.self_attn(encoder_sequences, encoder_sequences, encoder_sequences, key_padding_mask)

        encoder_sequences = self.self_attn_residual(encoder_sequences, self_attn)

        fcn, gating_variances = self.fcn(encoder_sequences)

        encoder_sequences = self.fcn_residual(encoder_sequences, fcn)
            
        return encoder_sequences, gating_variances

class Encoder(nn.Module):
    def __init__(self, args, vocab_size, norm=nn.LayerNorm):
        super(Encoder, self).__init__()

        self.args = args

        self.d_model = args.d_model

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.apply_dropout = nn.Dropout(args.dropout)
        self.norm = norm(args.d_model, args.norm_eps)
        self.encoder_layers = self.make_encoder_layers(args.n_encoder_layers, args.encoder_param_sharing_type, args.m_encoder_independent_layers, norm=norm)

        if args.positional_encoding_type != 'rotary':
            self.tensor_positional_encoding = nn.Parameter(utils.get_positional_encoding(args))

    def make_encoder_layers(self, n_layers, param_sharing_type, m_independent_layers, norm=nn.LayerNorm):
        def new_encoder_layer():
            return EncoderLayer(self.args, norm=norm)

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(new_encoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(new_encoder_layer())
                else:
                    layers.append(layers[i - 1])
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_encoder_layer())
        return nn.ModuleList(layers)

    def perform_embedding_transformation(self, encoder_sequences):
        return self.embedding(encoder_sequences) * math.sqrt(self.args.d_model) # (N, pad_length, d_model)

    def apply_positional_embedding(self, encoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return encoder_sequences + self.tensor_positional_encoding[:, :encoder_sequences.size(1), :]
        return encoder_sequences
    
    def apply_encoder_layer(self, encoder_sequences, key_padding_mask, encoder_layer):
        return encoder_layer(encoder_sequences, key_padding_mask)

    def forward(self, encoder_sequences, key_padding_mask):
        encoder_sequences = self.perform_embedding_transformation(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_positional_embedding(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_dropout(encoder_sequences) # (N, pad_length, d_model)

        gating_variances = []
        for encoder_layer in self.encoder_layers:
            encoder_sequences, gating_variance = self.apply_encoder_layer(encoder_sequences, key_padding_mask, encoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        # post-LN
        encoder_sequences = self.norm(encoder_sequences) # (N, pad_length, d_model)

        return encoder_sequences, gating_variances

class DecoderLayer(nn.Module):
    def __init__(self, args, norm=nn.LayerNorm):
        super(DecoderLayer, self).__init__()

        self.args = args

        self.self_attn = MultiHeadAttention(args, self_attn=True, in_decoder=True, norm=norm)
        self.cross_attn = MultiHeadAttention(args, self_attn=False, in_decoder=True, norm=norm)

        if args.use_admin:
            self.self_attn_residual = admin_torch.as_module(args.n_decoder_layers)
            self.cross_attn_residual = admin_torch.as_module(args.n_decoder_layers)
            self.fcn_residual = admin_torch.as_module(args.n_decoder_layers)
        else:
            self.self_attn_residual = Sum()
            self.cross_attn_residual = Sum()
            self.fcn_residual = Sum()

        self.fcn = PositionWiseFCNetwork(args, norm=norm)

    def forward(self, decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask):
        self_attn, _ = self.self_attn(decoder_sequences, decoder_sequences, decoder_sequences, tgt_key_padding_mask)
        decoder_sequences = self.self_attn_residual(decoder_sequences, self_attn)

        cross_attn, _ = self.cross_attn(decoder_sequences, encoder_sequences, encoder_sequences, src_key_padding_mask)
        decoder_sequences = self.cross_attn_residual(decoder_sequences, cross_attn)

        fcn, gating_variances = self.fcn(decoder_sequences)
        
        decoder_sequences = self.fcn_residual(decoder_sequences, fcn)

        return decoder_sequences, gating_variances

class Decoder(nn.Module):
    def __init__(self, args, vocab_size, norm=nn.LayerNorm):
        super(Decoder, self).__init__()

        self.args = args

        self.d_model = args.d_model

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.apply_dropout = nn.Dropout(args.dropout)
        self.layer_norm = norm(args.d_model, args.norm_eps)
        self.decoder_layers = self.make_decoder_layers(args.n_decoder_layers, args.decoder_param_sharing_type, args.m_decoder_independent_layers, norm=norm)
        self.classifier = nn.Linear(args.d_model, vocab_size)

        if args.positional_encoding_type != 'rotary':
            self.tensor_positional_encoding = nn.Parameter(utils.get_positional_encoding(args))

    def make_decoder_layers(self, n_layers, param_sharing_type, m_independent_layers, norm=nn.LayerNorm):
        def new_decoder_layer():
            return DecoderLayer(self.args, norm=norm)
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(new_decoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(new_decoder_layer())
                else:
                    layers.append(layers[i - 1])
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.cross_attn = layers[res_idx].cross_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    if hasattr(layers[res_idx], 'cross_attn_residual'):
                        new_layer.cross_attn_residual = layers[res_idx].cross_attn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.cross_attn = layers[res_idx].cross_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    if hasattr(layers[res_idx], 'cross_attn_residual'):
                        new_layer.cross_attn_residual = layers[res_idx].cross_attn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_decoder_layer())
        return nn.ModuleList(layers)

    def apply_embedding_transformation(self, decoder_sequences):
        return self.embedding(decoder_sequences) * math.sqrt(self.d_model) # (N, pad_length, d_model)
    
    def apply_positional_embedding(self, decoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return decoder_sequences + self.tensor_positional_encoding[:, :decoder_sequences.size(1), :]
        return decoder_sequences
    
    def apply_decoder_layer(self, decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask, decoder_layer):
        return decoder_layer(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask)

    def forward(self, decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask):
        decoder_sequences = self.apply_embedding_transformation(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_positional_embedding(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_dropout(decoder_sequences)

        gating_variances = []
        for decoder_layer in self.decoder_layers:
            decoder_sequences, gating_variance = self.apply_decoder_layer(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask, decoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        decoder_sequences = self.layer_norm(decoder_sequences)  # (N, pad_length, d_model)
        decoder_sequences = self.classifier(decoder_sequences)  # (N, pad_length, vocab_size)

        return decoder_sequences, gating_variances

class Transformer(nn.Module):
    def __init__(self, args, total_vocab_size, padding_value=0, norm=nn.LayerNorm):
        super(Transformer, self).__init__()
        self.args = args
        self.maxlen = args.maxlen
        self.padding_value = padding_value

        self.encoder = Encoder(args, total_vocab_size, norm=norm)
        self.decoder = Decoder(args, total_vocab_size, norm=norm)

    def forward(self, encoder_sequences, decoder_sequences, src_key_padding_mask, tgt_key_padding_mask):
        encoder_sequences, encoder_gating_variances = self.encoder(encoder_sequences, src_key_padding_mask) # (N, encoder_sequence_pad_length, d_model)
        decoder_sequences, decoder_gating_variances = self.decoder(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask) # (N, decoder_sequence_pad_length, vocab_size)
        return decoder_sequences, encoder_gating_variances, decoder_gating_variances
