import torch
import torch.nn as nn
import os
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from summarizer import make_summarizer
import numpy as np
import random

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")



def read_all_sequences():
    seqs = []
    files = os.listdir("dataset/Amazon_few_shot")
    root_dir = "dataset/Amazon_few_shot/"
    print("Read all sequences begin=====")
    for file in tqdm.tqdm(files):
        if "list" in file:
            continue
        path = os.path.join(root_dir, file)
        with open(path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        for line in lines:
            text, label = line.split("\t")
            text = text.strip()
            seqs.append(text)
    random.shuffle(seqs)
    return seqs

def create_pos_samples(args, seqs,summarizer):
    print("create pos samples begin====")
    num_iter = len(seqs)
    if args.toy:
        num_iter = args.toy_size
    sums = []
    for i in tqdm.tqdm(range(num_iter)):
        sum_seq = summarizer.sum_text(seqs[i])
        sums.append(sum_seq)

    return sums

def create_negative_samples(args, seqs, summarizer):
    neg_samples = []
    print("create neg samples begin====")
    num_iter = len(seqs)
    if args.toy:
        num_iter = args.toy_size
    for i in tqdm.tqdm(range(num_iter)):
        neg = []
        id_pool = np.arange(len(seqs))
        np.delete(id_pool,i)
        neg_ids = np.random.choice(id_pool, args.num_neg, replace=False)
        for id in neg_ids:
            neg.append(summarizer.sum_text(seqs[id]))
        neg_samples.append(neg)

    return neg_samples

def create_neutual_samples(args, seqs, summarizer):
    neutual_samples = []
    print("create neutual samples begin====")
    num_iter = len(seqs)
    if args.toy:
        num_iter = args.toy_size
    for i in tqdm.tqdm(range(num_iter)):
        neutual = []
        id_pool = np.arange(len(seqs))
        np.delete(id_pool,i)
        neutual_ids = np.random.choice(id_pool, args.num_neg, replace=False)
        for id in neutual_ids:
            neutual.append(summarizer.sum_text(seqs[i] + " " + seqs[id]))
        neutual_samples.append(neutual)

    return neutual_samples


def make_dataloader(args):
    summarizer = make_summarizer(args.summary_method)

    anchor_seqs = read_all_sequences()
    pos_seqs = create_pos_samples(args, anchor_seqs, summarizer)
    neg_seqs = create_negative_samples(args, anchor_seqs, summarizer)
    neutral_seqs = create_neutual_samples(args, anchor_seqs, summarizer)
    if args.toy:
        anchor_seqs = anchor_seqs[:args.toy_size]

    print("begin to make anchor ids=====")
    anchor_seqs_ids = tokenizer(anchor_seqs, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
    print("begin to make pos seqs ids=====")
    pos_seqs_ids = tokenizer(pos_seqs, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
    print("begin to make neg seqs ids=====")
    neg_seqs_ids = torch.LongTensor(anchor_seqs_ids.shape[0],args.num_neg,200)
    for i,neg_seq in enumerate(neg_seqs):
        neg_seqs_ids[i] = tokenizer(neg_seq, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]

    print("begin to make neg neutral ids=====")
    neutral_seqs_ids = torch.LongTensor(anchor_seqs_ids.shape[0],args.num_neg,200)
    for i,neutral_seq in enumerate(neutral_seqs):
        neutral_seqs_ids[i] = tokenizer(neutral_seq, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]

    dataset = TensorDataset(anchor_seqs_ids, pos_seqs_ids, neg_seqs_ids, neutral_seqs_ids)
    data_loader = DataLoader(dataset=dataset,batch_size=args.batch_size, shuffle=True,num_workers=2)

    return data_loader

    # print("Anchor======",anchor_seqs[0])
    # print("Pos======",pos_seqs[0])
    # print("Neg======",neg_seqs[0])
    # print("Neu======",neutral_seqs[0])
    # # print(anchor_seqs[0],pos_seqs[0],neg_seqs[0],neutral_seqs[0])





     


def make_train_loader(args):
    files = os.listdir("dataset/Amazon_few_shot")
    root_dir = "dataset/Amazon_few_shot/"
    token_ids = []
    summarizer = make_summarizer(args.summary_method)
    for file in tqdm.tqdm(files):
        if "list" in file:
            continue
        path = os.path.join(root_dir, file)
        with open(path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        for i, line in enumerate(lines):
            text, label = line.split("\t")
            text = text.strip()
            sum_text = summarizer.sum_text(text)
            ids = tokenizer.encode(text,padding=True,truncation=True, max_length=200)
            sum_ids = tokenizer.encode(sum_text,padding=True,truncation=True, max_length=200)
            ids += [1] * (200 - len(ids))
            token_ids.append(ids)
    dataset = torch.LongTensor(token_ids)
    dataset = TensorDataset(dataset)
    data_loader = DataLoader(dataset=dataset,batch_size=32, shuffle=True,num_workers=2)
    return data_loader
