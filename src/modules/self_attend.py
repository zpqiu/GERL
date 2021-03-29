# -*- encoding:utf-8 -*-
"""
编码title，得到语义表示
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttendLayer(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(SelfAttendLayer, self).__init__()

        self.trans_layer = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh()
        )
        self.gate_layer = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_num, embedding_size]
        :param seq_lens: shape [batch_size, seq_num]
        :return: shape [batch_size, embedding_size]
        """
        gates = self.gate_layer(self.trans_layer(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output
