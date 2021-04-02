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
            cfg.model.word_embedding_size, 
            heads_num=cfg.model.transformer.heads_num, 
            head_size=cfg.model.transformer.head_size,
        )
        mh_out_size = cfg.model.transformer.heads_num * cfg.model.transformer.head_size
        self.word_self_attend = SelfAttendLayer(mh_out_size, mh_out_size)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(
            self,
            seqs,
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
        embs = self.dropout(embs)
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

    def __init__(self, input_size, heads_num, head_size):
        super(MultiHeadedAttention, self).__init__()
        self.input_size = input_size
        self.heads_num = heads_num
        self.per_head_size = head_size

        self.query_proj = nn.Linear(input_size, heads_num * head_size)
        self.key_proj = nn.Linear(input_size, heads_num * head_size)
        self.value_proj = nn.Linear(input_size, heads_num * head_size)

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
        batch_size, seq_length, input_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, -1)

        query = self.query_proj(query).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
        # query = query.repeat(1, 1, heads_num).view(batch_size, -1, heads_num, input_size).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2)

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
