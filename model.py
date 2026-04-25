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

# =====================================================================
# Model v19 (Lite): Sequence-Level Cross-Attention + Task Query Decoder
# Parameter Count: ~19.6 Million
# =====================================================================

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
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )
    def forward(self, x):
        w = x.mean(dim=-1)
        return x * self.fc(w).unsqueeze(-1)

class EnhancedCNN1D(nn.Module):
    # REDUCED conv_dim from 256 to 128
    def __init__(self, vocab_size=33, embed_dim=128, conv_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.branches = nn.ModuleList()
        for k in [3, 5, 7]:
            self.branches.append(nn.Sequential(
                nn.Conv1d(embed_dim, conv_dim, k, padding=k//2),
                nn.BatchNorm1d(conv_dim), nn.GELU(),
                nn.Conv1d(conv_dim, conv_dim, k, padding=k//2),
                nn.BatchNorm1d(conv_dim), nn.GELU(),
            ))
        self.res_proj = nn.Conv1d(embed_dim, conv_dim, 1)
        self.se = SEBlock(conv_dim * 3)
        self.dropout = nn.Dropout(0.2)
        self.hidden_size = conv_dim * 3 * 2  # Now 768 instead of 1536

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        res = self.res_proj(x)
        outs = [branch(x) + res for branch in self.branches]
        combined = torch.cat(outs, dim=1)
        combined = self.se(combined)
        return self.dropout(torch.cat([combined.max(dim=-1).values, combined.mean(dim=-1)], dim=1))


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
        cnn_feat = self.cnn(seq_input)                          

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