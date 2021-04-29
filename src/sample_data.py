import pandas as pd
import random
import argparse
import torch
import re
from data import get_clean_line

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--seed', type=int, default=41)
args = parser.parse_args()

random.seed(args.seed)

dataset = args.dataset
path = "raw_datasets/" + dataset + "/train.csv"
train_pd = pd.read_csv(path)
train_data_np = train_pd.values
random.shuffle(train_data_np)
num_labels = 5
size = 16
if dataset == "ag_news":
  num_labels = 4
  size = 20

if dataset == "yahoo":
  num_labels = 10
  size = 16

lines = {i:[] for i in range(num_labels)}

text_idx = 2
if dataset == "yelp":
  text_idx = 1
for line in train_data_np:
  text = line[text_idx]
  if type(text) == float or (len(text.split()) < 10):
    continue
  text = text.strip()
  text = text.replace("\n"," ")
  text += "\n"
  label = line[0] - 1
  lines[label].append(text)

train_data = []
labels = []
for label in range(num_labels):
  labels += [label] * size
  random.shuffle(lines[label])
  train_data += lines[label][:size]




path = "raw_datasets/" + dataset + "/test.csv"
test_pd = pd.read_csv(path)
test_data_np = test_pd.values
random.shuffle(test_data_np)
num_labels = 5
size = 200
if dataset == "ag_news":
  num_labels = 4
  size = 250

if dataset == "yahoo":
  num_labels = 10
  size = 100
  
lines = {i:[] for i in range(num_labels)}

for line in test_data_np:
  text = line[text_idx]
  if type(text) == float or (len(text.split()) < 10):
    continue
  text = text.strip()
  text = text.replace("\n"," ")
  text += "\n"
  label = line[0] - 1
  lines[label].append(text)

test_data = []
test_labels = []
for label in range(num_labels):
  test_labels += [label] * size
  random.shuffle(lines[label])
  test_data += lines[label][:size]

clean_train_data = [get_clean_line(item)+'\n' for item in train_data]
clean_test_data = [get_clean_line(item)+'\n' for item in test_data]


PreSumm_input_path = "PreSumm/raw_data/" + dataset + "-seed-" + str(args.seed) + ".txt"
with open(PreSumm_input_path,"w") as fin:
  fin.writelines(clean_train_data)

with open("processed_data/"+dataset+"/seed-{}/train/data".format(int(args.seed)),"w") as fin:
  fin.writelines(clean_train_data)

with open("processed_data/"+dataset+"/seed-{}/test/data".format(int(args.seed)),"w") as fin:
  fin.writelines(clean_test_data)

torch.save(labels,"processed_data/"+dataset+"/seed-{}/train/labels".format(int(args.seed)))
torch.save(test_labels,"processed_data/"+dataset+"/seed-{}/test/labels".format(int(args.seed)))
print("sampled data saved on processed_data/"+dataset+"/seed-{}".format(int(args.seed))+"...")
