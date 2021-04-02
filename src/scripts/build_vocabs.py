# -*- coding: utf-8 -*-
"""
构建dict：
- news_id => news index
- user_id => user index
- word => word index
- news_index => title seq

news_id 和 user_id ，word 只用出现在train中的
"""
import os
import json
import pickle
import argparse

import pandas as pd
import numpy as np

from datasets.vocab import WordVocab
from utils.build_util import word_tokenize

ROOT_PATH = os.environ["GERL"]

def build_word_embeddings(vocab, pretrained_embedding, weights_output_file):
    # Load Glove embedding
    lines = open(pretrained_embedding, "r", encoding="utf8").readlines()
    emb_dict = dict()
    error_line = 0
    embed_size = 0
    for line in lines:
        row = line.strip().split()
        try:
            embedding = [float(w) for w in row[1:]]
            emb_dict[row[0]] = np.array(embedding)
            if embed_size == 0:
                embed_size = len(embedding)
        except:
            error_line += 1
    print("Error lines: {}".format(error_line))

    # embed_size = len(emb_dict.values()[0])
    # build embedding weights for model
    weights_matrix = np.zeros((len(vocab), embed_size))
    words_found = 0

    for i, word in enumerate(vocab.itos):
        try:
            weights_matrix[i] = emb_dict[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(size=(embed_size,))
    print("Totally find {} words in pre-trained embeddings.".format(words_found))
    np.save(weights_output_file, weights_matrix)

def build_user_id_vocab(cfg, behavior_df):
    user_vocab = WordVocab(behavior_df.uid.values, max_size=1000000, min_freq=1, lower=cfg.lower)
    print("USER ID VOCAB SIZE: {}".format(len(user_vocab)))
    f_user_vocab_path = os.path.join(ROOT_PATH, cfg.output, "userid_vocab.bin")
    user_vocab.save_vocab(f_user_vocab_path)
    return user_vocab

def build_news_id_vocab(cfg, news_df):
    news_vocab = WordVocab(news_df.newsid.values, max_size=140000, min_freq=1, lower=cfg.lower)
    print("NEWS ID VOCAB SIZE: {}".format(len(news_vocab)))
    f_news_vocab_path = os.path.join(ROOT_PATH, cfg.output, "newsid_vocab.bin")
    news_vocab.save_vocab(f_news_vocab_path)
    return news_vocab

def build_word_vocab(cfg, news_df):
    word_vocab = WordVocab(news_df.title_token.values, max_size=cfg.size, min_freq=1, lower=cfg.lower)
    print("TEXT VOCAB SIZE: {}".format(len(word_vocab)))
    f_text_vocab_path = os.path.join(ROOT_PATH, cfg.output, "word_vocab.bin")
    word_vocab.save_vocab(f_text_vocab_path)
    return word_vocab

def build_newsid_to_title(cfg, news_df: pd.DataFrame, newsid_vocab: WordVocab, word_vocab: WordVocab):
    news2title = np.zeros((len(newsid_vocab) + 1, cfg.max_title_len), dtype=int)
    news2title[0] = word_vocab.to_seq('<pad>', seq_len=cfg.max_title_len)
    for row in news_df[["newsid", "title_token"]].values:
        news_id, title = row[:2]
        news_index = newsid_vocab.stoi[news_id]
        news2title[news_index], cur_len = word_vocab.to_seq(title, seq_len=cfg.max_title_len, with_len=True)
    
    f_title_matrix = os.path.join(ROOT_PATH, "data", cfg.fsize, "news_title.npy")
    np.save(f_title_matrix, news2title)
    print("title embedding: ", news2title.shape)

def main(cfg):
    # Build vocab
    print("Loading news info")
    f_train_news = os.path.join(ROOT_PATH, "data", cfg.fsize, "train/news.tsv")
    f_dev_news = os.path.join(ROOT_PATH, "data", cfg.fsize, "dev/news.tsv")
    f_test_news = os.path.join(ROOT_PATH, "data", cfg.fsize, "test/news.tsv")

    print("Loading training news")
    train_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                           names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                           quoting=3)
    if os.path.exists(f_dev_news):
        print("Loading dev news")
        dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                               names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                               quoting=3)
    if os.path.exists(f_test_news):
        print("Loading testing news")
        test_news = pd.read_csv(f_test_news, sep="\t", encoding="utf-8",
                                names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                                quoting=3)

    all_news = pd.concat([train_news, dev_news, test_news], ignore_index=True)
    all_news = all_news.drop_duplicates("newsid")
    print("All news: {}".format(len(all_news)))

    # 单独处理train news
    train_news['title_token'] = train_news['title'].apply(lambda x: ' '.join(word_tokenize(x)))
    all_news['title_token'] = all_news['title'].apply(lambda x: ' '.join(word_tokenize(x)))

    # Build user id vocab
    f_train_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "train/behaviors.tsv")
    f_dev_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "dev/behaviors.tsv")
    f_test_behaviors = os.path.join(ROOT_PATH, "data", cfg.fsize, "test/behaviors.tsv")
    train_behavior = pd.read_csv(f_train_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    dev_behavior = pd.read_csv(f_dev_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    test_behavior = pd.read_csv(f_test_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    behaviors_df = pd.concat([train_behavior, dev_behavior, test_behavior], ignore_index=True)
    behaviors_df = behaviors_df.drop_duplicates("uid")
    _ = build_user_id_vocab(cfg, behaviors_df)
    
    # Build news id vocab
    newsid_vocab = build_news_id_vocab(cfg, all_news)

    # Build word vocab
    word_vocab = build_word_vocab(cfg, train_news)

    # Build word embeddings
    print("Building word embedding matrix")
    pretrain_path = os.path.join(ROOT_PATH, cfg.pretrain)
    weight_path = os.path.join(ROOT_PATH, cfg.output, "word_embeddings.bin")
    build_word_embeddings(word_vocab, pretrain_path, weight_path)

    # Build news_index => title word seq
    build_newsid_to_title(cfg, all_news, newsid_vocab, word_vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--fsize", default="L", type=str,
                        help="Corpus size")
    parser.add_argument("--pretrain", default="data/glove.840B.300d.txt", type=str,
                        help="Path of the raw review data file.")
    parser.add_argument("--output", default="data/vocabs", type=str,
                        help="Path of the training data file.")
    parser.add_argument("--size", default=80000, type=int,
                        help="Path of the validation data file.")
    parser.add_argument("--max_title_len", default=20, type=int,
                        help="Path of the validation data file.")
    parser.add_argument("--lower", action='store_true')

    args = parser.parse_args()

    main(args)
