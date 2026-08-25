import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torchvision.transforms import Resize
from ast import literal_eval
from sklearn.metrics import matthews_corrcoef
from torchvision import models
from transformers import AutoTokenizer, AutoModelForMaskedLM
import math
import json
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LogNorm
import random



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM


class ESM2_Encoder(nn.Module):
    def __init__(self, model_name, trainable=True, unfreeze_last_n=0):
        super().__init__()
        self.esm_mlm = AutoModelForMaskedLM.from_pretrained(model_name)
        self.hidden_size = self.esm_mlm.config.hidden_size
        if not trainable:
            for param in self.esm_mlm.parameters():
                param.requires_grad = False
            if unfreeze_last_n > 0:
                for layer in self.esm_mlm.esm.encoder.layer[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        return self.esm_mlm.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

class SEBlock(nn.Module):
    """
    Mask-aware Squeeze-and-Excitation block.

    Input:
        x        : [B, C, L]
        seq_mask : [B, L], 1 for valid tokens, 0 for PAD

    The channel descriptor is computed using only valid sequence positions.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x, seq_mask):
        # [B, L] -> [B, 1, L]
        mask = seq_mask.unsqueeze(1).to(dtype=x.dtype)

        # Remove PAD contributions
        x_masked = x * mask

        # Number of valid positions per sequence
        lengths = seq_mask.sum(
            dim=1,
            keepdim=True
        ).clamp_min(1).to(dtype=x.dtype)

        # Masked global average pooling over sequence dimension
        # [B, C, L] -> [B, C]
        channel_descriptor = (
            x_masked.sum(dim=-1) / lengths
        )

        # Channel-wise gates
        gates = self.fc(channel_descriptor).unsqueeze(-1)

        # Apply channel recalibration
        return x * gates

class ConvBlock(nn.Module):
    """
    Mask-aware two-layer 1D convolution block.

    Uses LayerNorm instead of BatchNorm so that padded positions
    do not participate in batch/sequence normalization statistics.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.norm1 = nn.LayerNorm(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.norm2 = nn.LayerNorm(out_channels)

    @staticmethod
    def apply_layernorm(x, norm):
        """
        Conv output: [B, C, L]
        LayerNorm expects normalized dimension at the end.
        """
        x = x.transpose(1, 2)      # [B, L, C]
        x = norm(x)
        x = x.transpose(1, 2)      # [B, C, L]
        return x

    def forward(self, x, mask):
        """
        x    : [B, C, L]
        mask : [B, 1, L]
        """
        # First convolution
        y = self.conv1(x)
        y = self.apply_layernorm(
            y,
            self.norm1
        )

        y = F.gelu(y)

        # Explicitly remove PAD activations
        y = y * mask

        # Second convolution
        y = self.conv2(y)

        y = self.apply_layernorm(
            y,
            self.norm2
        )

        y = F.gelu(y)

        # Explicitly remove PAD activations again
        y = y * mask

        return y

class EnhancedCNN1D(nn.Module):
    """
    Mask-aware multi-scale CNN for peptide/protein sequences.

    Branches:
        kernel 3 -> effective receptive field 5
        kernel 5 -> effective receptive field 9
        kernel 7 -> effective receptive field 13

    Each branch contains two convolutional layers.

    Output:
        [B, 2 * 3 * conv_dim]

    For conv_dim=128:
        output = [B, 768]
    """

    def __init__(
        self,
        vocab_size=33,
        embed_dim=128,
        conv_dim=128
    ):
        super().__init__()


        # Token embedding
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=1 # ESM TOKENIZER
        )

        # Multi-scale CNN branches
        self.branches = nn.ModuleList([
            ConvBlock(
                in_channels=embed_dim,
                out_channels=conv_dim,
                kernel_size=k
            )
            for k in [3, 5, 7]
        ])

        # Residual projection
        self.res_proj = nn.Conv1d(
            embed_dim,
            conv_dim,
            kernel_size=1,
            bias=False
        )

        # Squeeze-and-Excitation
        self.se = SEBlock(
            channels=conv_dim * 3,
            reduction=4
        )

        self.dropout = nn.Dropout(0.2)

        # Three branches × conv_dim channels
        # Max pooling + mean pooling
        self.hidden_size = conv_dim * 3 * 2

    def forward(self, x, seq_mask):
        """
        Args:
            x:
                [B, L] token IDs

            seq_mask:
                [B, L]
                1 = valid token
                0 = PAD
        Returns:
            CNN feature vector:
                [B, hidden_size]

        For conv_dim=128:
            [B, 768]
        """


        # Validate mask


        if seq_mask is None:
            raise ValueError(
                "seq_mask must be provided to EnhancedCNN1D. "
                "CNN padding masking must never be bypassed."
            )
        # Embedding
        # [B, L] -> [B, L, embed_dim]
        x = self.embed(x)

        # [B, L, embed_dim] -> [B, embed_dim, L]
        x = x.transpose(1, 2)

        # [B, L] -> [B, 1, L]
        mask = seq_mask.unsqueeze(1).to(dtype=x.dtype)

        # Explicitly zero PAD embeddings
        x = x * mask

        # Residual pathway
        res = self.res_proj(x)

        # Remove PAD activations
        res = res * mask

        # Multi-scale branches
        outs = []

        for branch in self.branches:
            y = branch(x, mask)
            # Residual connection
            y = y + res
            # Guarantee PAD = 0 after residual addition
            y = y * mask
            outs.append(y)

        # Concatenate branches
        # [B, 128*3, L]
        combined = torch.cat(
            outs,
            dim=1
        )

        # Guarantee no PAD signal
        combined = combined * mask

        # Mask-aware SE
        combined = self.se(
            combined,
            seq_mask
        )

        # SE can theoretically produce nonzero values
        # at PAD positions, so mask once more.
        combined = combined * mask

        # MASKED GLOBAL MAX POOLING
        # PAD cannot become the maximum.
        combined_for_max = combined.masked_fill(
            seq_mask.unsqueeze(1) == 0,
            torch.finfo(combined.dtype).min
        )

        max_pooled = combined_for_max.max(
            dim=-1
        ).values


        # MASKED GLOBAL MEAN POOLING
        combined_for_mean = combined * mask

        # Actual number of valid tokens
        lengths = seq_mask.sum(
            dim=1,
            keepdim=True
        ).clamp_min(1).to(
            dtype=combined.dtype
        )

        mean_pooled = (
            combined_for_mean.sum(dim=-1)
            / lengths
        )

        # FINAL REPRESENTATION
        pooled = torch.cat(
            [
                max_pooled,
                mean_pooled
            ],
            dim=1
        )

        return self.dropout(pooled)


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.Sequential(
            nn.Linear(dim, 128), 
            nn.Tanh(), 
            nn.Linear(128, num_heads)
        )
        self._last_weights = None

    def forward(self, x, mask):
        scores = self.attn(x)
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        weights = torch.softmax(scores, dim=1)
        self._last_weights = weights
        pooled = (x.unsqueeze(2) * weights.unsqueeze(-1)).sum(dim=1)
        return pooled.view(x.size(0), -1)


    def orthogonality_loss(self):
        """Penalize overlap between head attention distributions."""
        if self._last_weights is None:
            return 0.0
            
        # weights: [B, L, num_heads] -> transpose to [B, num_heads, L]
        w = self._last_weights.transpose(1, 2)  
        
        # Gram matrix of head attention distributions: shape [B, num_heads, num_heads]
        gram = torch.bmm(w, w.transpose(1, 2))  
        
        # Create a boolean mask for the off-diagonal elements (~torch.eye inverts the identity matrix)
        mask = ~torch.eye(self.num_heads, dtype=torch.bool, device=gram.device)
        
        # Penalize only the off-diagonal overlap to encourage heads to focus on different tokens.
        # We want these dot products to be pushed towards 0.
        loss = (gram[:, mask] ** 2).mean()
        
        return loss
    
class GatedFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x)

class PeptideNetwork(nn.Module):
    def __init__(self, num_classes=21, mask_token_id=32):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.num_classes = num_classes

        # BOTH encoders are now the 8M parameter t6 model
        self.esm_t6_a = ESM2_Encoder("facebook/esm2_t6_8M_UR50D", trainable=True)        
        self.esm_t6_b = ESM2_Encoder("facebook/esm2_t6_8M_UR50D", trainable=False, unfreeze_last_n=2)                                  
        self.cnn = EnhancedCNN1D()          # Bx768

        # REDUCED cross_dim to 128 to save parameters
        cross_dim = 128
        self.proj_t6_a = nn.Linear(320, cross_dim)
        self.proj_t6_b = nn.Linear(320, cross_dim) # Updated to 320 for t6

        self.cross_t6_a = nn.MultiheadAttention(cross_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.ln_ca_t6_a = nn.LayerNorm(cross_dim)
        
        self.cross_t6_b = nn.MultiheadAttention(cross_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.ln_ca_t6_b = nn.LayerNorm(cross_dim)
        
        self.pool_t6_a = MultiHeadAttentionPool(cross_dim, num_heads=4)
        self.pool_t6_b = MultiHeadAttentionPool(cross_dim, num_heads=4)

        # Bottleneck reduction layer before fusion
        # Concat size: 128*4*3 (pools) + 768 (CNN) = 1536 + 768 = 2304
        concat_size = cross_dim * 4 * 2 + self.cnn.hidden_size
        
        self.dim_reduce = nn.Sequential(
            nn.Linear(concat_size, 512),
            nn.GELU()
        )
        
        # Fusion now operates efficiently on 512 dimensions
        self.fusion = GatedFusion(512)
        self.ln = nn.LayerNorm(512)

        # binary head
        binary_features_dim = 64
        self.binary_features = nn.Sequential(
            nn.Linear(512, binary_features_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.binary_classifier = nn.Linear(64, 1) # Final logit

        # Task Query Decoder
        self.task_dim = 128
        self.n_memory_tokens = 16
        self.memory_proj = nn.Sequential(
            nn.Linear(512 + binary_features_dim, self.task_dim * self.n_memory_tokens),
            nn.GELU(),
        )
        self.task_queries = nn.Embedding(num_classes, self.task_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.task_dim, nhead=4, dim_feedforward=256,
            batch_first=True, dropout=0.1
        )
        self.task_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.task_classifiers = nn.ModuleList([
            nn.Linear(self.task_dim, 1) for _ in range(num_classes)
        ])

    def _mask_tokens(self, input_ids, attention_mask, mask_prob=0.15):
        masked_ids = input_ids.clone()
        prob_matrix = torch.full_like(input_ids, mask_prob, dtype=torch.float)
        prob_matrix[attention_mask == 0] = 0
        prob_matrix[:, 0] = 0
        seq_lens = attention_mask.sum(dim=1)
        for i in range(len(seq_lens)):
            if seq_lens[i] > 1:
                prob_matrix[i, seq_lens[i] - 1] = 0
        mask = torch.bernoulli(prob_matrix).bool()
        masked_ids[mask] = self.mask_token_id
        return masked_ids

    def _extract_features(self, seq_input, seq_mask):
        esm6_a_seq = self.esm_t6_a(seq_input, seq_mask)       
        esm6_b_seq = self.esm_t6_b(seq_input, seq_mask)     
        cnn_feat = self.cnn(seq_input, seq_mask)                            

        t6_a = self.proj_t6_a(esm6_a_seq)       
        t6_b = self.proj_t6_b(esm6_b_seq)    


        kv_pad = seq_mask == 0  

        ca_t6_a, _ = self.cross_t6_a(t6_a, t6_b, t6_b, key_padding_mask=kv_pad)
        ca_t6_a = self.ln_ca_t6_a(t6_a + ca_t6_a)

        ca_t6_b, _ = self.cross_t6_b(t6_b, t6_a, t6_a, key_padding_mask=kv_pad)
        ca_t6_b = self.ln_ca_t6_b(t6_b + ca_t6_b)

        pooled_t6_a = self.pool_t6_a(ca_t6_a, seq_mask)
        pooled_t6_b = self.pool_t6_b(ca_t6_b, seq_mask)

        # Concat -> Reduce -> Fuse
        combined = torch.cat([pooled_t6_a, pooled_t6_b, cnn_feat], dim=1)  
        reduced = self.dim_reduce(combined)
        fusion = self.ln(reduced + self.fusion(reduced))

        binary_features = self.binary_features(fusion)
        
        return binary_features, torch.cat([fusion, binary_features], dim=1)
    
    def _binary_classify(self, binary_features):
        
        binary_logits = self.binary_classifier(binary_features)

        return binary_logits

    def _classify(self, final_fusion):
        B = final_fusion.size(0)
        memory = self.memory_proj(final_fusion).view(B, self.n_memory_tokens, self.task_dim)
        tgt = self.task_queries.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.task_decoder(tgt, memory)
        logits = torch.cat([self.task_classifiers[i](decoded[:, i, :])
                           for i in range(self.num_classes)], dim=1)
        return logits

    def forward(self, seq_input, seq_mask, mask_tokens=False):
        if mask_tokens and self.training:
            seq_input = self._mask_tokens(seq_input, seq_mask)
        
        binary_features, combined_features = self._extract_features(seq_input, seq_mask)
        

        return self._binary_classify(binary_features), self._classify(combined_features)


    def ortho_loss(self):
        return (self.pool_t6_a.orthogonality_loss() +
                self.pool_t6_b.orthogonality_loss() 
                ) / 2
    
    def get_features(self, seq_input, seq_mask, mask_tokens=False):
        if mask_tokens and self.training:
            seq_input = self._mask_tokens(seq_input, seq_mask)
        return self._extract_features(seq_input, seq_mask)

    def multi_classify(self, combined):
        return self._classify(combined)
    
    def binary_classify(self, binary_features):
        return self._binary_classify(binary_features)


    def _binary_classify(self, binary_features):
        
        binary_logits = self.binary_classifier(binary_features)

        return binary_logits

    def _classify(self, final_fusion):
        B = final_fusion.size(0)
        memory = self.memory_proj(final_fusion).view(B, self.n_memory_tokens, self.task_dim)
        tgt = self.task_queries.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.task_decoder(tgt, memory)
        logits = torch.cat([self.task_classifiers[i](decoded[:, i, :])
                           for i in range(self.num_classes)], dim=1)
        return logits

    def forward(self, seq_input, seq_mask, mask_tokens=False):
        if mask_tokens and self.training:
            seq_input = self._mask_tokens(seq_input, seq_mask)
        
        binary_features, combined_features = self._extract_features(seq_input, seq_mask)
        

        return self._binary_classify(binary_features), self._classify(combined_features)


    def ortho_loss(self):
        return (self.pool_t6_a.orthogonality_loss() +
                self.pool_t6_b.orthogonality_loss() 
                ) / 2
    
    def get_features(self, seq_input, seq_mask, mask_tokens=False):
        if mask_tokens and self.training:
            seq_input = self._mask_tokens(seq_input, seq_mask)
        return self._extract_features(seq_input, seq_mask)

    def multi_classify(self, combined):
        return self._classify(combined)
    
    def binary_classify(self, binary_features):
        return self._binary_classify(binary_features)


endpoints = ['anti-bacterial',
 'anti-cancer',
 'anti-fungal',
 'anti-parasitic',
 'anti-viral',
 'cell-cell-communication',
 'drug-delivery',
 'immunological',
 'inhibitor',
 'metabolic',
 'non-functional',
 'other-functional',
 'signal-peptide',
 'toxic']

 
endpoints_set = set(endpoints)
endpoint_index = {endpoint: i for i, endpoint in enumerate(endpoints)}
index_endpoint = {i: endpoint for i, endpoint in enumerate(endpoints)}
NON_FUNC_IDX = endpoint_index['non-functional']
FUNC_INDICES = [i for k, i in endpoint_index.items() if k != 'non-functional']
