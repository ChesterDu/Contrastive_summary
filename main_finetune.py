import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
import torch.backends.cudnn as cudnn
from models import EmbNetwork
from builder import MoCo
import data
import os
import tqdm
from opt import OpenAIAdam
from train import train
import tqdm


def evaluate_model(model, test_loader, log):
    print("Evaluation Start======")
    model.eval()
    TP, TN, FN, FP = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            seq_ids, labels = [item.to(device) for item in batch]
            logits = model(seq_ids,labels=labels)[1]
            # print(logits)

            prediction = torch.argmax(logits, dim = 1)
            TP += ((prediction == 1) & (labels == 1)).sum().item()
            # TN    predict 和 label 同时为0
            TN += ((prediction == 0) & (labels == 0)).sum().item()
            # FN    predict 0 label 1
            FN += ((prediction == 0) & (labels == 1)).sum().item()
            # FP    predict 1 label 0
            FP += ((prediction == 1) & (labels == 0)).sum().item()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("recall: ",r)
    print("precision: ",p)
    print("F1: ",F1)
    print("Acc: ",acc)

    log["recall"] = r
    log["precision"] = p
    log["F1"] = F1
    log["acc"] = acc

    return log




parser = argparse.ArgumentParser()
parser.add_argument("--use_pretrain", action='store_true')
parser.add_argument('--steps', type=int, default=10000)
parser.add_argument("--toy",action='store_true')
parser.add_argument("--local_rank",type=int, default=0)
parser.add_argument("--num_workers",type=int,default=1)
parser.add_argument("--toy_size",type=int,default=1000)
parser.add_argument("--seed",type=int,default=41)
parser.add_argument("--gpu_ids",type=int,default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len",type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clip',type=float,default=1)
parser.add_argument('--log_step',type=int,default=100)
parser.add_argument('--log_dir',type=str,default="finetune_log.pkl")
parser.add_argument('--model_dir',type=str,default="finetune_model.pkl")

args = parser.parse_args()


if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True


dataset = torch.load("finetune_dataset.pkl")
train_dataset = dataset["train"]
test_dataset = dataset["test"]
# train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, num_workers=2,batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, num_workers=2,batch_size=args.batch_size, shuffle=True, drop_last=True)

##make model
config = RobertaConfig.from_pretrained("roberta-base")
config.num_labels = 2
model = RobertaForSequenceClassification.from_pretrained("roberta-base",config=config)
## Load Pretrained Weight
if args.use_pretrain:
    pretained_weight = torch.load("checkpoint.pkl", map_location='cpu')
    for key in pretained_weight:
        pretained_weight[key] = pretained_weight[key].cpu()
    model_weight = model.state_dict()
    for key in pretained_weight:
        if "pooler" in key:
            continue
        new_key = key.replace("module.base_model","roberta")
        model_weight[new_key] = deepcopy(pretained_weight[key])

    model.load_state_dict(model_weight)

##make optimizer
optimizer = OpenAIAdam(model.parameters(),
                                  lr=args.lr,
                                  schedule='warmup_linear',
                                  warmup=0.002,
                                  t_total=args.steps,
                                  b1=0.9,
                                  b2=0.999,
                                  e=1e-08,
                                  l2=0.01,
                                  vector_l2=True,
                                  max_grad_norm=args.clip)

critirion = torch.nn.CrossEntropyLoss()
device = torch.device(args.gpu_ids)
model = model.to(device)

step = 0
bar = tqdm.tqdm(total=args.steps)
bar.update(0)
loss_list = []
best_acc = 0
logs = []

while(step < args.steps):
    for batch in train_loader:
        optimizer.zero_grad()
        seq_ids, labels = [item.to(device) for item in batch]
        loss = model(seq_ids,labels = labels)[0]
        # print(loss.item())

        # loss =critirion(logits,labels)
        
        loss.backward()
        loss_list.append(loss.item())

        optimizer.step()
        step += 1
        if (step % 10 == 0):
            bar.update(10)
        
        if (step % args.log_step == 0):
            
            print("step: ",step)
            print("loss: ",sum(loss_list)/step)
            log = {"step":step, "loss":sum(loss_list)/step}
            log = evaluate_model(model,test_loader,log)
            logs.append(log)
            torch.save(logs, args.log_dir)
            
            if (log["acc"] > best_acc):
                best_acc = log["acc"]
                torch.save(model.state_dict(),args.model_dir)
            model.train()











