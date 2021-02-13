import torch
from transformers import RobertaTokenizer
from summarizer import summarize
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
import random
class multiLabelDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_name,max_num,seed=41,split="train"):
        random.seed(seed)
        num_labels = 5
        if dataset_name is "ag_news":
          num_labels = 4

        num_per_class = int(max_num / num_labels)
        if split is "test":
          num_per_class = int(10000 / num_labels)
        

        temp = {i:[] for i in range(num_labels)}
        with open ("../processed_data/" + dataset_name + "/" + split + "/data") as fin:
          text = fin.readlines()

        labels = torch.load("../processed_data/" + dataset_name + "/" + split + "/labels")

        with open ("../processed_data/" + dataset_name + "/train/summary") as fin:
          summary = fin.readlines()
        
        if split is "test":
          summary = text

        for i in range(len(text)):
          x,y,s = text[i].strip(),labels[i],summary[i].strip()
          if split is "train":
            s = s.replace("<q>",". ")
            if "....." in s or "this this this" in s or "is is is" in s:
              s = summarize(text)

          temp[y].append((y,x,s))
        
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

    x_ids = tokenizer(x, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
    s_mix_ids = tokenizer(s_mix, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]

    return x_ids, s_mix_ids, y_a, y_b

def collate_fn(batch):
    batch_size = len(batch)
    perm_index = torch.randperm(batch_size)
    y_a = torch.LongTensor([item[0] for item in batch])
    # print(y_a)
    # print(batch)
    y_b = y_a[perm_index]
    x = [item[1] for item in batch]
    
    s = [item[2] for item in batch]

    x_ids = tokenizer(x, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
    s_ids = tokenizer(s, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]

    return x_ids, s_ids, y_a, y_b