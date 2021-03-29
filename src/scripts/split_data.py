# encoding: utf-8
"""
Split the big txt file to multiple parts
Date: 15 Mar, 2019
"""
import os
import random
import argparse

random.seed(7)
ROOT_PATH = os.environ["GERL"]

def build(args, input_path: str, out_path: str, shuffle=False):
    print("Loading...")
    f_samples = os.path.join(ROOT_PATH, "data", input_path)
    lines = open(f_samples, "r", encoding="utf8").readlines()
    if shuffle:
        random.shuffle(lines)

    print("There are {0} lines.".format(len(lines)))
    sub_file_length = len(lines) // args.num

    for i in range(args.num):
        output_file_path = os.path.join(ROOT_PATH, "data", "{}.p{}.tsv".format(out_path, i))
        st = i * sub_file_length
        if i == args.num-1:
            ed = len(lines)
        else:
            ed = st + sub_file_length
        print("Creating sub-file {0} ...".format(i))
        with open(output_file_path, "w", encoding="utf8") as fw:
            for j in range(st, ed):
                fw.write(lines[j])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num", type=int, default=7,
                        help="the number of slice. Default 7.")
    parser.add_argument("-i", "--input_path", default="L/examples/training_examples.tsv", type=str,
                        help="Path of the whole taining file needed to be cut")
    parser.add_argument("-o", "--output_path", default="L/examples/training_examples", type=str,
                        help="Path of the output files.")
    parser.add_argument("--shuffle", action='store_true')
    args = parser.parse_args()

    build(args, args.input_path, args.output_path)
