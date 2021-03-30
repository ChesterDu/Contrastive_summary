import pandas as pd
import random
import argparse
import torch

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
  print(dataset)
  num_labels = 4
  size = 20
lines = {i:[] for i in range(num_labels)}
for line in train_data_np:
  text = line[2]
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
size = 2000
if dataset is "ag_news":
  num_labels = 4
  size = 2500
lines = {i:[] for i in range(num_labels)}
for line in test_data_np:
  text = line[2]
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




PreSumm_input_path = "PreSumm/raw_data/" + dataset + ".txt"
with open(PreSumm_input_path,"w") as fin:
  fin.writelines(train_data)

with open("processed_data/"+dataset+"/train/data","w") as fin:
  fin.writelines(train_data)

with open("processed_data/"+dataset+"/test/data","w") as fin:
  fin.writelines(test_data)

torch.save(labels,"processed_data/"+dataset+"/train/labels")
torch.save(test_labels,"processed_data/"+dataset+"/test/labels")
print("sampled data saved on processed_data/"+dataset+"...")
