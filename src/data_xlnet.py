import torch
from transformers import XLNetTokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

def collate_fn_mix(batch):
    batch_size = len(batch)
    perm_index = torch.randperm(batch_size)
    y_a = torch.LongTensor([item[0] for item in batch])
    # print(y_a)
    # print(batch)
    y_b = y_a[perm_index]
    x = [item[1] for item in batch]
    # x_perm = [x[perm_index[i]] for i in range(batch_size)]
    # s = [summarize(x[i]) for i in range(batch_size)]
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
    # x_perm = [x[perm_index[i]] for i in range(batch_size)]
    # s = [summarize(x[i]) for i in range(batch_size)]
    s = [item[2] for item in batch]
    # s_perm = [s[perm_index[i]] for i in range(batch_size)]
    # s_mix = [s[i] + "\n" + s_perm[i] for i in range(batch_size)]

    x_ids = tokenizer(x, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
    s_ids = tokenizer(s, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]

    return x_ids, s_ids, y_a, y_b