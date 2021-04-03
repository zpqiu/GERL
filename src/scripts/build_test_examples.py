# -*- coding: utf-8 -*-
"""Script for building the test examples.
{
    imp_id: 000,
    user: 123,
    hist_news: [1, 2, 3]
    neighbor_users: [4, 5, 6]
    target_news: 7,
    y: 1
    neighbor_news: [27, 28, 29]
}
"""
import os
import json
import random
import argparse
from typing import List, Dict

import tqdm
import pandas as pd

from datasets.vocab import WordVocab

random.seed(7)
ROOT_PATH = os.environ["GERL"]


def build_examples(cfg, df: pd.DataFrame, 
                   user_vocab: WordVocab, newsid_vocab: WordVocab, 
                   user_one_hop: Dict, news_one_hop: Dict, 
                   user_two_hop: Dict, news_two_hop: Dict):
    
    def _get_neighors(neighbor_dict, key, max_neighbor_num):
        neighbors = neighbor_dict.get(key, [])
        return neighbors[:max_neighbor_num]

    f_out = os.path.join(ROOT_PATH, "data", cfg.fsize, "examples/test_examples.tsv")
    fw = open(f_out, "w", encoding="utf-8")

    random.seed(7)
    loader = tqdm.tqdm(df[["uid", "imp", "id"]].values, desc="Building")
    for row in loader:
        # row = json.loads(row)
        uid = row[0]
        user_index = user_vocab.stoi.get(uid, 0)
        hist_news = _get_neighors(user_one_hop, user_index, cfg.max_user_one_hop)
        neighbor_users = _get_neighors(user_two_hop, user_index, cfg.max_user_two_hop)
        
        samples = row[1].strip().split()
        for sample in samples:
            news_id = sample
            neighbor_news = []
            target_news = newsid_vocab.stoi.get(news_id, 0)
            neighbor_news = _get_neighors(news_two_hop, target_news, cfg.max_news_two_hop)
            
            j = {
                "user": user_index,
                "hist_news": hist_news,
                "neighbor_users": neighbor_users,
                "target_news": target_news,
                # "hist_users": hist_users,
                "neighbor_news": neighbor_news,
                "y": 0,
                "imp_id": row[-1],
            }
            fw.write(json.dumps(j) + "\n")


def load_hop_dict(fpath: str) -> Dict:
    lines = open(fpath, "r", encoding="utf-8").readlines()
    d = dict()
    error_line_count = 0
    for line in lines:
        row = line.strip().split("\t")
        if len(row) != 2:
            error_line_count += 1
            continue
        key, vals = row[:2]
        vals = [int(x) for x in vals.split(",")]
        d[int(key)] = vals
    print("{} error lines: {}".format(fpath, error_line_count))
    return d

def main(cfg):
    f_test_samples = os.path.join(ROOT_PATH, "data", cfg.fsize, "test/behaviors.tsv")
    f_news_vocab = os.path.join(ROOT_PATH, "data", cfg.fvocab, "newsid_vocab.bin")
    f_user_vocab = os.path.join(ROOT_PATH, "data", cfg.fvocab, "userid_vocab.bin")
    f_user_one_hop = os.path.join(ROOT_PATH, "data", cfg.fvocab, "test-user_one_hops.txt")
    f_news_one_hop = os.path.join(ROOT_PATH, "data", cfg.fvocab, "test-news_one_hops.txt")
    f_user_two_hop = os.path.join(ROOT_PATH, "data", cfg.fvocab, "test-user_two_hops.txt")
    f_news_two_hop = os.path.join(ROOT_PATH, "data", cfg.fvocab, "test-news_two_hops.txt")

    # Load vocab
    user_vocab = WordVocab.load_vocab(f_user_vocab)
    newsid_vocab = WordVocab.load_vocab(f_news_vocab)
    user_one_hop = load_hop_dict(f_user_one_hop)
    news_one_hop = load_hop_dict(f_news_one_hop)
    user_two_hop = load_hop_dict(f_user_two_hop)
    news_two_hop = load_hop_dict(f_news_two_hop)

    # 预处理好的训练样本
    samples = pd.read_csv(f_test_samples, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])

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
