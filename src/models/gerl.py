# -*- encoding:utf-8 -*-
"""
GERL 的主模型

{
    user: 123,
    hist_news: [1, 2, 3]
    neighbor_users: [4, 5, 6]
    target_news: [7, 8, 9, 10, 11],
    neighbor_news: [
        [27, 28, 29],
        [30, 31, 32],
        [33, 34, 35],
        [36, 37, 38],
        [39, 40, 41]
    ]
}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.self_attend import SelfAttendLayer
from modules.title_encoder import TitleEncoder


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # Config
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.neg_count = cfg.model.neg_count
        self.embedding_size = cfg.model.id_embedding_size
        self.max_user_one_hop = cfg.dataset.max_user_one_hop
        self.max_user_two_hop = cfg.dataset.max_user_two_hop
        self.max_news_two_hop = cfg.dataset.max_news_two_hop

        # Init Layers
        self.user_embedding = nn.Embedding(cfg.dataset.user_count, cfg.model.id_embedding_size)
        self.newsid_embedding = nn.Embedding(cfg.dataset.news_count, cfg.model.id_embedding_size)
        self.title_encoder = TitleEncoder(cfg)
        self.user_two_hop_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.user_one_hop_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.news_two_hop_id_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.news_two_hop_title_attend = SelfAttendLayer(cfg.model.id_embedding_size, cfg.model.id_embedding_size)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def _arrange_input(self, batch):
        user, hist_news, neighbor_users = batch["user"], batch["hist_news"], batch["neighbor_users"]
        target_news, neighbor_news = batch["target_news"], batch["neighbor_news"]
        
        return user, hist_news, neighbor_users, target_news, neighbor_news

    def forward(self, user, hist_news, neighbor_users, target_news, neighbor_news, target_news_cnt):
        """
        user: [*]
        hist_news: [*, max_user_one_hop]
        neighbor_users: [*, max_user_two_hop]
        target_news: [*, target_news_cnt]
        neighbor_news: [*, target_news_cnt, max_news_two_hop]

        return [*, target_news_cnt]
        """
        user_embedding = self.user_embedding(user)
        neighbor_users_embedding = self.user_embedding(neighbor_users)
        neighbor_news_embedding = self.newsid_embedding(neighbor_news)

        # User
        user_two_hop_rep = self.user_two_hop_attend(neighbor_users_embedding)
        hist_news = hist_news.view(-1)
        hist_news_reps = self.title_encoder(hist_news)
        hist_news_reps = hist_news_reps.view(-1, self.max_user_one_hop, self.embedding_size)
        user_one_hop_rep = self.user_one_hop_attend(hist_news_reps)

        # News
        target_news = target_news.view(-1)
        target_news_reps = self.title_encoder(target_news)
        target_news_reps = target_news_reps.view(-1, target_news_cnt, self.embedding_size)
        
        neighbor_news_embedding = neighbor_news_embedding.view(-1, self.max_news_two_hop, self.embedding_size)
        news_two_hop_id_reps = self.news_two_hop_id_attend(neighbor_news_embedding)
        news_two_hop_id_reps = news_two_hop_id_reps.view(-1, target_news_cnt, self.embedding_size)

        neighbor_news = neighbor_news.view(-1)
        neighbor_news_reps = self.title_encoder(neighbor_news)
        neighbor_news_reps = neighbor_news_reps.view(-1, self.max_news_two_hop, self.embedding_size)
        news_two_hop_title_reps = self.news_two_hop_title_attend(neighbor_news_reps)
        news_two_hop_title_reps = news_two_hop_title_reps.view(-1, target_news_cnt, self.embedding_size)

        # Logit
        final_user_rep = torch.cat([user_one_hop_rep, user_embedding, user_two_hop_rep], dim=-1)
        final_user_rep = final_user_rep.repeat(1, target_news_cnt).view(-1, self.embedding_size * 3)
        final_target_reps = torch.cat([news_two_hop_title_reps, news_two_hop_id_reps, target_news_reps])
        final_target_reps = final_target_reps.view(-1, self.embedding_size * 3)

        logits = torch.sum(final_user_rep * final_target_reps, dim=-1)
        logits = logits.view(-1, target_news_cnt)

        return logits

    def training_step(self, batch_data):
        # REQUIRED
        user, hist_news, neighbor_users, target_news, neighbor_news = self._arrange_input(batch_data)
        logits = self.forward(user, hist_news, neighbor_users, target_news, neighbor_news, 5)

        target = batch_data["y"]
        loss = F.cross_entropy(logits, target)

        return loss

    def prediction_step(self, batch_data):
        user, hist_news, neighbor_users, target_news, neighbor_news = self._arrange_input(batch_data)
        target_news = target_news.unsqueeze(-1)
        neighbor_news = neighbor_news.unsqueeze(1)
        logits = self.forward(user, hist_news, neighbor_users, target_news, neighbor_news, 1)
        return logits.view(-1)
