# -*- coding: utf-8 -*-
"""Script for building the training examples.

"""
import os
import json
import random
import argparse
from typing import List, Dict

import tqdm

from datasets.vocab import WordVocab

random.seed(7)
ROOT_PATH = os.environ["GERL"]


def build_examples(cfg, df: List[str], 
                   user_vocab: WordVocab, newsid_vocab: WordVocab, 
                   user_one_hop: Dict, news_one_hop: Dict, 
                   user_two_hop: Dict, news_two_hop: Dict):
    
    def _get_neighors(neighbor_dict, key, max_neighbor_num):
        neighbors = neighbor_dict.get(key, [])
        return neighbors[:max_neighbor_num] + [0, ] * (max_neighbor_num - len(neighbors))

    f_out = os.path.join(ROOT_PATH, "data", cfg.fsize, "examples/training_examples.tsv")
    fw = open(f_out, "w", encoding="utf-8")

    random.seed(7)
    loader = tqdm.tqdm(df, desc="Building")
    for row in loader:
        row = json.loads(row)
        uid = row["uid"]
        user_index = user_vocab.stoi[uid]
        hist_news = _get_neighors(user_one_hop, user_index, cfg.max_user_one_hop)
        neighbor_users = _get_neighors(user_two_hop, user_index, cfg.max_user_two_hop)
        
        for pair in row["pairs"]:
            hist_users = []
            neighbor_news = []
            target_news = [newsid_vocab.stoi.get(pair[0], 0)] + [newsid_vocab.stoi.get(x, 0) for x in pair[1]]
            for news_index in target_news:
                hist_users.append(_get_neighors(news_one_hop, news_index, cfg.max_news_one_hop))
                neighbor_news.append(_get_neighors(news_two_hop, news_index, cfg.max_news_two_hop))
            j = {
                "user": user_index,
                "hist_news": hist_news,
                "neighbor_users": neighbor_users,
                "target_news": target_news,
                "hist_users": hist_users,
                "neighbor_news": neighbor_news
            }
            fw.write(json.dumps(j) + "\n")


def load_hop_dict(fpath: str) -> Dict:
    lines = open(fpath, "r", encoding="utf-8").readlines()
    d = dict()
    for line in lines:
        key, vals = line.split("\t")[:2]
        vals = [int(x) for x in vals.split(",")]
        d[key] = vals


def main(cfg):
    f_train_samples = os.path.join(ROOT_PATH, "data", cfg.fsize, "train/samples.tsv")
    f_news_vocab = os.path.join(ROOT_PATH, "data", cfg.fvocab, "newsid_vocab.bin")
    f_user_vocab = os.path.join(ROOT_PATH, "data", cfg.fvocab, "userid_vocab.bin")
    f_user_one_hop = os.path.join(ROOT_PATH, "data", cfg.fvocab, "train-user_one_hops.txt")
    f_news_one_hop = os.path.join(ROOT_PATH, "data", cfg.fvocab, "train-news_one_hops.txt")
    f_user_two_hop = os.path.join(ROOT_PATH, "data", cfg.fvocab, "train-user_two_hops.txt")
    f_news_two_hop = os.path.join(ROOT_PATH, "data", cfg.fvocab, "train-news_two_hops.txt")

    # Load vocab
    user_vocab = WordVocab.load_vocab(f_user_vocab)
    newsid_vocab = WordVocab.load_vocab(f_news_vocab)
    user_one_hop = load_hop_dict(f_user_one_hop)
    news_one_hop = load_hop_dict(f_news_one_hop)
    user_two_hop = load_hop_dict(f_user_two_hop)
    news_two_hop = load_hop_dict(f_news_two_hop)

    # 预处理好的训练样本
    samples = open(f_train_samples, "r", encoding="utf-8").readlines()

    build_examples(cfg, samples, user_vocab, newsid_vocab, user_one_hop, news_one_hop, user_two_hop, news_two_hop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsize", default="L", type=str,
                        help="Corpus size")
    parser.add_argument("--fout", default="hop1_cocur_bip_hist50", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--fvocab", default="vocabs", type=str,
                        help="Path of the output dir.")

    args = parser.parse_args()

    main(args)
