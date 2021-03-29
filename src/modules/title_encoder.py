# -*- encoding:utf-8 -*-
"""
编码title，得到语义表示
"""
import math

import numpy as np
import torch
import torch.nn as nn

from datasets.vocab import WordVocab
from modules.self_attend import SelfAttendLayer
from utils.model_util import build_embedding_layer


class TitleEncoder(nn.Module):
    def __init__(self, cfg):
        super(TitleEncoder, self).__init__()
        self.cfg = cfg
        self.max_hist_len = cfg.dataset.max_hist_len
        self.batch_size = cfg.training.batch_size
        self.hidden_size = cfg.model.hidden_size

        self.vocab = WordVocab.load_vocab(cfg.dataset.word_vocab)
        self.word_embedding = build_embedding_layer(
            pretrained_embedding_path=cfg.dataset.get("word_embedding", ""),
            vocab=self.vocab,
            embedding_dim=cfg.model.word_embedding_size,
        )
        # map news_id => word seq
        self.title_embedding = torch.LongTensor(np.load(cfg.dataset.title_embedding))

        # NRMS hidden size is 150
        self.mh_self_attn = MultiHeadedAttention(
            self.hidden_size, heads_num=cfg.model.transformer.heads_num
        )
        self.word_self_attend = SelfAttendLayer(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forword(
            self,
            seqs,
            seq_lens,
    ):
        """

        Args:
            seqs: [*]
            seq_lens: [*]

        Returns:
            [*, hidden_size]
        """
        # [*, max_title_len]
        titles = self.title_embedding[seqs]
        # [*, max_title_len, hidden_size]
        embs = self.word_embedding(titles)
        hiddens = self.mh_self_attn(embs, embs, embs)
        hiddens = self.dropout(hiddens)

        # [*, hidden_size]
        self_attend_reps = self.word_self_attend(hiddens)

        return self_attend_reps


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(2)
        ])
        self.final_linear = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, key, value, query, mask=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        orginal_value = value
        key, value = [l(x). \
                        view(batch_size, -1, heads_num, per_head_size). \
                        transpose(1, 2) \
                    for l, x in zip(self.linear_layers, (key, value))
                    ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        if mask is not None:
            scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        # probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))

        return output


def create_mask_from_lengths_for_seqs(
        seq_lens: torch.Tensor, max_len: int
) -> torch.Tensor:
    """
    :param seq_lens: shape [batch_size, ]
    :param max_len: int
    :return: shape [batch_size, seq_length]
    """
    segs = torch.arange(max_len, device=seq_lens.device, dtype=seq_lens.dtype).expand(
        len(seq_lens), max_len
    ) < seq_lens.unsqueeze(1)

    return segs.long()
