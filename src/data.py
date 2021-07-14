import torch
from transformers import RobertaTokenizer,BertTokenizer,XLNetTokenizer
from summarizer import summarize
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
import random
import re

def make_tokenizer(args):
  if args.model == "xlnet":
    return XLNetTokenizer.from_pretrained("xlnet-base-cased")
  
  if args.model == "bert":
    return BertTokenizer.from_pretrained("bert-base-uncased")
  
  if args.model == "roberta":
    return RobertaTokenizer.from_pretrained("roberta-base")
  

def get_clean_line(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm,.?!;':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    if clean_line[-1] == " ":
        clean_line = clean_line[:-1]
    return clean_line

class multiLabelDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_name,max_num,seed=41,split="train",aug_methods="summary"):
        random.seed(seed)
        num_labels = 5
        if dataset_name == "imdb":
          num_labels = 2
        if dataset_name == "ag_news":
          num_labels = 4
        print(dataset_name)
        if dataset_name == "yahoo":
          print("here")
          num_labels = 10

        num_per_class = int(max_num / num_labels)
        if dataset_name == "yahoo":
          num_per_class = 16
        if split == "test":
          num_per_class = int(1000 / num_labels)
        

        temp = {i:[] for i in range(num_labels)}
        with open ("../processed_data/" + dataset_name + "/seed-{}/".format(int(seed)) + split + "/data") as fin:
          text = fin.readlines()

        labels = torch.load("../processed_data/" + dataset_name + "/seed-{}/".format(int(seed)) + split + "/labels")

        
        
        if split=="train" and "+" in aug_methods:
          text += text
          labels += labels
          with open ("../processed_data/" + dataset_name + "/seed-{}/".format(int(seed)) + "train/" + "summary") as fin:
            aug_data = fin.readlines()
          with open ("../processed_data/" + dataset_name + "/seed-{}/".format(int(seed)) + "train/" + "eda") as fin:
            aug_data += fin.readlines()
        if split=="train" and "+" not in aug_methods:
          with open ("../processed_data/" + dataset_name + "/seed-{}/".format(int(seed)) + "train/" + aug_methods) as fin:
            aug_data = fin.readlines()

        if split is "test":
          aug_data = text

        print(len(text))
        print(len(labels))
        print(len(aug_data))
        for i in range(len(text)):
          x,y,s = text[i].strip(),labels[i],aug_data[i].strip()
          if (split == "train") and (aug_methods == "summary"):
            s = s.replace("<q>",". ")
            if "....." in s or "this this this" in s or "is is is" in s:
              s = summarize(text)

          temp[y].append((y,get_clean_line(x),get_clean_line(s)))
        
        self.data = []
        for i in range(num_labels):
          random.shuffle(temp[i])
          self.data += temp[i][:num_per_class]
        random.shuffle(self.data)

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
        


def collate_fn_mix(batch):
    batch_size = len(batch)
    perm_index = torch.randperm(batch_size)
    y_a = torch.LongTensor([item[0] for item in batch])
    y_b = y_a[perm_index]
    x = [item[1] for item in batch]
    s = [item[2] for item in batch]
    s_perm = [s[perm_index[i]] for i in range(batch_size)]
    s_mix = [s[i] + "\n" + s_perm[i] for i in range(batch_size)]

    # x_ids = tokenizer(x, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
    # s_mix_ids = tokenizer(s_mix, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]

    return x, s_mix, y_a, y_b

def collate_fn_feature_mix(batch):
  batch_size = len(batch)
  perm_index = torch.randperm(batch_size)
  y_a = torch.LongTensor([item[0] for item in batch])
  y_b = y_a[perm_index]
  x = [item[1] for item in batch]
  s = [item[2] for item in batch]
  s_perm = [s[perm_index[i]] for i in range(batch_size)]
  x_perm = [x[perm_index[i]] for i in range(batch_size)]
  

  return x,x_perm,s,s_perm,y_a,y_b

def collate_fn(batch):
    batch_size = len(batch)
    perm_index = torch.randperm(batch_size)
    y_a = torch.LongTensor([item[0] for item in batch])
    # print(y_a)
    # print(batch)
    y_b = y_a[perm_index]
    x = [item[1] for item in batch]
    
    s = [item[2] for item in batch]

    # x_ids = tokenizer(x, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
    # s_ids = tokenizer(s, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]

    return x, s, y_a, y_b