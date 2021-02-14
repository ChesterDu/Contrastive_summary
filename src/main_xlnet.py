import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import XLNetModel, XLNetConfig, XLNetForSequenceClassification
from data_xlnet import collate_fn, collate_fn_mix
from data import multiLabelDataset
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
import torch.backends.cudnn as cudnn
import os
import tqdm
from opt import OpenAIAdam
import tqdm


class Recoder_multi():
    def __init__(self,args):
        self.args = args
        self.loss = []
        self.acc = []
        self.step = []
    def log_train(self,loss):
        self.loss.append(loss.item())
    
    def log_test(self,acc,step):
        self.acc.append(acc)
        self.step.append(step)


    def meter(self,step):
        st,ed = step - self.args.log_step, step
        print("===================================")
        print("step: ",step)
        print("loss: ",sum(self.loss[st:ed])/self.args.log_step)
    



def evaluate_model(model, test_loader, recoder, step):
    print("Evaluation Start======")
    model.eval()

    # bar = tqdm.tqdm(total=len(test_loader))
    # bar.update(0)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x_ids, s_mix_ids, y_a, y_b = batch
            seq_ids = x_ids.to(device)
            labels = y_a.to(device)
            logits = model(input_ids=seq_ids,labels=labels)[1]

            prediction = torch.argmax(logits, dim = 1)
            correct += (prediction == labels).sum().item()
            total += prediction.shape[0]

    acc = correct / total
    print("Acc: ",acc)

    recoder.log_test(acc,step)

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=10000)
parser.add_argument("--seed",type=int,default=41)
parser.add_argument("--gpu_ids",type=int,default=0)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip',type=float,default=1)
parser.add_argument('--log_step',type=int,default=100)
parser.add_argument('--log_dir',type=str,default="finetune_log.pkl")

parser.add_argument('--dataset',type=str,default="amazon")
parser.add_argument('--train_num',type=float,default=80)

args = parser.parse_args()

# Make Dataset
train_dataset = multiLabelDataset(dataset_name = args.dataset,max_num=args.train_num,seed=args.seed,split="train")
test_dataset = multiLabelDataset(dataset_name = args.dataset,max_num=10000,seed=args.seed,split="test")



my_collect = collate_fn
train_loader = DataLoader(train_dataset, num_workers=2, batch_size=args.batch_size, shuffle=True, collate_fn = my_collect)
test_loader = DataLoader(test_dataset, num_workers=2, batch_size=args.eval_batch_size,shuffle=False,collate_fn=my_collect)

# ##make model
device = torch.device(args.gpu_ids)
config = XLNetConfig.from_pretrained("xlnet-base-cased")
config.num_labels = args.num_labels
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",config=config)
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

# critirion = torch.nn.CrossEntropyLoss()

model = model.to(device)

step = 0
bar = tqdm.tqdm(total=args.steps)
bar.update(0)
best_acc = 0
recoder = Recoder_multi(args)
loss_mask_str = args.loss_mask.split(',')
loss_mask = [float(i) for i in loss_mask_str]
print(loss_mask)

best_loss = float('inf')
count = 0
begin_eval = False
while(step < args.steps):
    model.train()
    for batch in train_loader:
        x_ids, s_mix_ids, y_a, y_b = batch
        seq_ids = x_ids.to(device)
        labels = y_a.to(device)
        loss = model(input_ids=seq_ids,labels=labels)[0]
        # print(ce_loss_x, ce_loss_s, scl_loss, ucl_loss)
        loss.backward()

        count += 1
        if (count % args.num_accum == 0):
            optimizer.step()
            recoder.log_train(ce_loss_x, ce_loss_s, scl_loss,ucl_loss,loss)
            step += 1
            optimizer.zero_grad()

            if (step >= args.steps):
                break

            if (step % args.log_step == 0):
                begin_eval = True

            if (step % 10 == 0):
                bar.update(10)
        
        
        if begin_eval:
            recoder.meter(step)
            evaluate_model(model,test_loader,recoder,step)
            torch.save(recoder, "../"+args.log_dir)

            model.train()
            begin_eval = False













