# -*- coding: utf-8 -*-
"""
构建两个1-hop关系表
- news index => user index list, 可以提前做好sample
- user index => news index list, 可以提前做好sample

构建两个 2-hop关系表
- news index => 2-hop news index list, 可以提前做好sample
- user index => 2-hop user index list, 可以提前做好sample

只能利用 hist 信息构建这个表，只保留在train中出现过的 user 和 item
train 和 val的时候共用上边的表，都是基于train的hist来构建
test 单独跑，基于trian + test 的hist来构建
"""
import os
import json
import random
import argparse
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from datasets.vocab import WordVocab

ROOT_PATH = os.environ["GERL"]

def build_two_hop_neighbors(cfg, user_one_hop: Dict, news_one_hop: Dict, part: str):
    user_dict = dict()
    news_dict = dict()
    for user, news_list in tqdm(user_one_hop.items(), desc="Building hop-2 user"):
        two_hop_users = []
        for news in news_list:
            two_hop_users += news_one_hop[news]
        if len(two_hop_users) > cfg.max_user_two_hop:
            two_hop_users = random.sample(two_hop_users, cfg.max_user_two_hop)
        user_dict[user] = two_hop_users
    for news, user_list in tqdm(news_one_hop.items(), desc="Building hop-2 news"):
        two_hop_news = []
        for user in user_list:
            two_hop_news += user_one_hop[user]
        if len(two_hop_news) > cfg.max_news_two_hop:
            two_hop_news = random.sample(two_hop_news, cfg.max_news_two_hop)
        news_dict[news] = two_hop_news
    
    f_user = os.path.join(ROOT_PATH, "data", cfg.fvocab, "{}-user_two_hops.txt".format(part))
    f_news = os.path.join(ROOT_PATH, "data", cfg.fvocab, "{}-news_two_hops.txt".format(part))
    with open(f_user, "w", encoding="utf-8") as fw:
        for user, news_list in user_dict.items():
            news_list_str = ",".join([str(x) for x in news_list])
            fw.write("{}\t{}\n".format(user, news_list_str))
    with open(f_news, "w", encoding="utf-8") as fw:
        for news, user_list in news_dict.items():
            user_list_str = ",".join([str(x) for x in user_list])
            fw.write("{}\t{}\n".format(news, user_list_str))


def build_one_hop_neighbors(cfg, behavior_df: pd.DataFrame, user_vocab: WordVocab, newsid_vocab: WordVocab, part: str) -> Tuple[Dict, Dict]:
    behavior_df = behavior_df.fillna("")
    user_dict = dict()
    news_dict = dict()

    for uid, hist in tqdm(behavior_df[["uid", "hist"]].values, desc="Building Hop-1"):
        if uid not in user_vocab.stoi:
            continue
        user_index = user_vocab.stoi[uid]

        if user_index not in user_dict:
            user_dict[user_index] = []

        for newsid in hist.split():
            newsid = newsid.strip()
            if newsid not in newsid_vocab.stoi:
                continue
            news_index = newsid_vocab.stoi[newsid]
            if news_index not in news_dict:
                news_dict[news_index] = []
            # click_list.append([user_index, news_index])
            if len(user_dict[user_index]) < cfg.max_user_one_hop:
                user_dict[user_index].append(news_index)
            if len(news_dict[news_index]) < cfg.max_news_one_hop:
                news_dict[news_index].append(user_index)
    
    f_user = os.path.join(ROOT_PATH, "data", cfg.fvocab, "{}-user_one_hops.txt".format(part))
    f_news = os.path.join(ROOT_PATH, "data", cfg.fvocab, "{}-news_one_hops.txt".format(part))
    with open(f_user, "w", encoding="utf-8") as fw:
        for user, news_list in user_dict.items():
            news_list_str = ",".join([str(x) for x in news_list[:cfg.max_user_one_hop]])
            fw.write("{}\t{}\n".format(user, news_list_str))
    with open(f_news, "w", encoding="utf-8") as fw:
        for news, user_list in news_dict.items():
            user_list_str = ",".join([str(x) for x in user_list[:cfg.max_news_one_hop]])
            fw.write("{}\t{}\n".format(news, user_list_str))

    return user_dict, news_dict


def main(cfg):
    f_train_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "train/behaviors.tsv")
    f_news_vocab = os.path.join(ROOT_PATH, "data", cfg.fvocab, "newsid_vocab.bin")
    f_user_vocab = os.path.join(ROOT_PATH, "data", cfg.fvocab, "userid_vocab.bin")

    # Load vocab
    user_vocab = WordVocab.load_vocab(f_user_vocab)
    newsid_vocab = WordVocab.load_vocab(f_news_vocab)

    train_behavior = pd.read_csv(f_train_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    train_behavior = train_behavior.fillna("")
    train_behavior = train_behavior[train_behavior["hist"]!=""].drop_duplicates("uid")
    f_test_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "test/behaviors.tsv")
    test_behavior = pd.read_csv(f_test_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    test_behavior = test_behavior.fillna("")
    test_behavior = test_behavior[test_behavior["hist"]!=""].drop_duplicates("uid")

    # Train one- and two-hops
    train_user_one_hop, train_news_one_hop = build_one_hop_neighbors(cfg, train_behavior, user_vocab, newsid_vocab, "train")
    build_two_hop_neighbors(cfg, train_user_one_hop, train_news_one_hop, "train")

    # Test one- and two-hops
    test_user_one_hop, test_news_one_hop = build_one_hop_neighbors(cfg, test_behavior, user_vocab, newsid_vocab, "test")
    build_two_hop_neighbors(cfg, test_user_one_hop, test_news_one_hop, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="L", type=str,
                        help="Corpus size")
    parser.add_argument("--fvocab", default="vocabs", type=str,
                        help="Path of the training data file.")
    parser.add_argument("--max_user_one_hop", default=50, type=int,
                        help="Maximum number of user one-hop neighbors.")
    parser.add_argument("--max_news_one_hop", default=50, type=int,
                        help="Maximum number of news one-hop neighbors.")
    parser.add_argument("--max_user_two_hop", default=15, type=int,
                        help="Maximum number of user two-hop neighbors.")
    parser.add_argument("--max_news_two_hop", default=15, type=int,
                        help="Maximum number of news two-hop neighbors.")

    args = parser.parse_args()

    main(args)
