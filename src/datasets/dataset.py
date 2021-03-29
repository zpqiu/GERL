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
import json

import tqdm
import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, args, corpus_path):
        self.args = args
        self.max_user_two_hop = args.max_user_two_hop
        self.max_user_one_hop = args.max_user_one_hop
        self.max_news_two_hop = args.max_news_two_hop
        self.corpus_path = corpus_path

        with open(corpus_path, "r") as f:
            self.lines = [line.strip()
                          for line in tqdm.tqdm(f, desc="Loading Dataset")]
            self.corpus_lines = len(self.lines)
    
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        user, hist_news, neighbor_users, target_news, neighbor_news = self.parse_line(item)

        if len(hist_news) < self.max_user_one_hop:
            hist_news = hist_news + [0, ] * (self.max_user_one_hop - len(hist_news))

        if len(neighbor_users) < self.max_user_two_hop:
            neighbor_users = neighbor_users + [0, ] * (self.max_user_two_hop - len(neighbor_users))

        for idx in range(len(neighbor_news)):
            if len(neighbor_news[idx]) < self.max_news_two_hop:
                neighbor_news[idx] = neighbor_news[idx] + [0, ] * (self.max_news_two_hop - len(neighbor_news[idx]))

        output = {"user": user,
                  "hist_news": hist_news,
                  "neighbor_users": neighbor_users,
                  "target_news": target_news,
                  "neighbor_news": neighbor_news,
                  "y": 0
                }

        return {key: torch.tensor(value) for key, value in output.items()}

    def parse_line(self, item):
        line = self.lines[item]
        j = json.loads(line)

        user = j["user"]
        hist_news = j["hist_news"]
        neighbor_users = j["neighbor_users"]
        target_news = j["target_news"]
        neighbor_news = j["neighbor_news"]

        return user, hist_news, neighbor_users, target_news, neighbor_news


class ValidationDataset(TrainingDataset):
    def __getitem__(self, item):
        user, hist_news, neighbor_users, target_news, neighbor_news, y, imp_id = self.parse_line(item)

        if len(hist_news) < self.max_user_one_hop:
            hist_news = hist_news + [0, ] * (self.max_user_one_hop - len(hist_news))

        if len(neighbor_users) < self.max_user_two_hop:
            neighbor_users = neighbor_users + [0, ] * (self.max_user_two_hop - len(neighbor_users))

        if len(neighbor_news) < self.max_news_two_hop:
            neighbor_news = neighbor_news + [0, ] * (self.max_news_two_hop - len(neighbor_news))

        output = {"user": user,
                  "hist_news": hist_news,
                  "neighbor_users": neighbor_users,
                  "target_news": target_news,
                  "neighbor_news": neighbor_news,
                  "y": y,
                  "imp_id": imp_id,
                }

        return {key: torch.tensor(value) for key, value in output.items()}

    def parse_line(self, item):
        line = self.lines[item]
        j = json.loads(line)

        user = j["user"]
        hist_news = j["hist_news"]
        neighbor_users = j["neighbor_users"]
        target_news = j["target_news"]
        neighbor_news = j["neighbor_news"]
        y = j["y"]
        imp_id = j["imp_id"]

        return user, hist_news, neighbor_users, target_news, neighbor_news, y, imp_id
