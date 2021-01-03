import torch
import torch.nn as nn
import os
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from summarizer import SummarizerWrap
import numpy as np
import random

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

is_test = 0
is_dev = 1
is_train = 2

def data_type(file_name):
    test_list = ['books','dvd', 'electronics', 'kitchen']
    for test_name in test_list:
        if test_name in file_name:
            if 'test' in file_name:
                return is_test
            if 'dev' in file_name:
                return is_dev

    return is_train
    
def read_file(path,seqs):
    with open(path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    for line in lines:
        text, label = line.split("\t")
        text = text.strip()
        label = int(label.strip())
        if label == -1:
            label = 0
        seqs.append((text,label))


def read_all_sequences(args):
    train_seqs = []
    test_seqs = []

    files = os.listdir("dataset/Amazon_few_shot")
    root_dir = "dataset/Amazon_few_shot/"
    print("Read all sequences begin=====")
    for file in tqdm.tqdm(files):
        if "list" in file:
            continue
        path = os.path.join(root_dir, file)
        split = data_type(file)
        if (split == is_test):
            read_file(path,test_seqs)
        elif(split == is_dev): 
            continue
        elif(split == is_train):
            read_file(path,train_seqs)
        
    # random.shuffle(seqs)
    if args.toy:
        train_seqs = train_seqs[:args.toy_size]
        test_seqs = test_seqs[:args.toy_size]
    return train_seqs,test_seqs



def make_dataset(args,seqs):
    seq_num = len(seqs)
    max_len = args.max_len
    seq_ids = []
    seq_labels = torch.LongTensor(seq_num)
    for i,(seq, label) in enumerate(seqs):
        seq_ids.append(seq)
        seq_labels[i] = label
    seq_ids = tokenizer(seq_ids, padding = 'max_length', max_length = max_len, truncation = True, return_tensors="pt")["input_ids"]
    dataset = TensorDataset(seq_ids, seq_labels)

    return dataset
