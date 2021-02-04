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
import scl_data
import os
import tqdm
from opt import OpenAIAdam
from train import train
import tqdm
from scl_model import scl_model

class Recoder():
    def __init__(self,args):
        self.args = args
        self.ce_loss_x = []
        self.ce_loss_s = []
        self.ce_loss_semi = []
        self.scl_loss = []
        self.scl_loss_semi = []
        self.loss = []
        self.p = []
        self.r = []
        self.f1 = []
        self.acc = []
        self.step = []
    def log_train(self,ce_loss_x, ce_loss_s, ce_loss_semi, scl_loss, scl_loss_semi,loss):
        self.ce_loss_x.append(ce_loss_x.item())
        self.ce_loss_s.append(ce_loss_s.item())
        self.ce_loss_semi.append(ce_loss_semi.item())
        self.scl_loss.append(scl_loss.item())
        self.scl_loss_semi.append(scl_loss_semi.item())
        self.loss.append(loss.item())
    
    def log_test(self,p,r,f1,acc,step):
        self.p.append(p)
        self.r.append(p)
        self.f1.append(f1)
        self.acc.append(f1)
        self.step.append(step)



    def meter(self,step):
        print("===================================")
        print("loss: ",sum(self.loss)/step)
        print("ce_loss_x: ",sum(self.ce_loss_x)/step)
        print("ce_loss_s: ",sum(self.ce_loss_s)/step)
        print("ce_loss_semi: ",sum(self.ce_loss_semi)/step)
        print("scl_loss: ",sum(self.scl_loss)/step)
        print("scl_loss_semi: ",sum(self.scl_loss_semi)/step)



class Recoder_multi():
    def __init__(self,args):
        self.args = args
        self.ce_loss_x = []
        self.ce_loss_mix = []
        self.scl_loss = []
        self.loss = []
        self.acc = []
        self.step = []
    def log_train(self,ce_loss_x, ce_loss_mix, scl_loss,loss):
        self.ce_loss_x.append(ce_loss_x.item())
        self.ce_loss_mix.append(ce_loss_mix.item())
        self.scl_loss.append(scl_loss.item())
        self.loss.append(loss.item())
    
    def log_test(self,acc,step):
        self.acc.append(f1)
        self.step.append(step)


    def meter(self,step):
        print("===================================")
        print("loss: ",sum(self.loss)/step)
        print("ce_loss_x: ",sum(self.ce_loss_x)/step)
        print("ce_loss_mix: ",sum(self.ce_loss_mix)/step)
        print("scl_loss: ",sum(self.scl_loss)/step)


def evaluate_model(model, test_loader, recoder, step, binary = True):
    print("Evaluation Start======")
    model.eval()
    TP, TN, FN, FP = 0, 0, 0, 0
    
    step = 0
    bar = tqdm.tqdm(total=len(test_loader))
    bar.update(0)
    with torch.no_grad():
        for batch in test_loader:
            seq_ids, labels = [item.to(device) for item in batch]
            logits = model.predict(seq_ids)
            # print(logits)

            prediction = torch.argmax(logits, dim = 1)
            TP += ((prediction == 1) & (labels == 1)).sum().item()
            # TN    predict 和 label 同时为0
            TN += ((prediction == 0) & (labels == 0)).sum().item()
            # FN    predict 0 label 1
            FN += ((prediction == 0) & (labels == 1)).sum().item()
            # FP    predict 1 label 0
            FP += ((prediction == 1) & (labels == 0)).sum().item()

            step += 1
            # if (step % 40 == 0):
                # bar.update(40)

    # p = TP / (TP + FP)
    # r = TP / (TP + FN)
    # F1 = 2 * r * p / (r + p)
    # acc = (TP + TN) / (TP + TN + FP + FN)
    p = TP / (TP + FP) if (TP + FP) != 0 else 0
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p) if (r + p) != 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("recall: ",r)
    print("precision: ",p)
    print("F1: ",F1)
    print("Acc: ",acc)

    recoder.log_test(p,r,F1,acc,step)




parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=10000)
parser.add_argument("--toy",action='store_true')
parser.add_argument("--local_rank",type=int, default=0)
parser.add_argument("--num_workers",type=int,default=1)
parser.add_argument("--toy_size",type=int,default=1000)
parser.add_argument("--seed",type=int,default=41)
parser.add_argument("--gpu_ids",type=int,default=0)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--num_accum",type=int,default=1)
parser.add_argument("--max_len",type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip',type=float,default=1)
parser.add_argument('--t_pos',type=float,default=0.9)
parser.add_argument('--t_mix',type=float,default=0.9)
parser.add_argument('--trade_off',type=float,default=0.5)
parser.add_argument("--lambd",type=float,default=0.8)


parser.add_argument('--log_step',type=int,default=100)
parser.add_argument('--log_dir',type=str,default="finetune_log.pkl")
parser.add_argument('--model_dir',type=str,default="finetune_model.pkl")

parser.add_argument('--dataset_pth',type=str,default="none")
parser.add_argument('--dataset',type=str,default="amazon_2")
parser.add_argument('--percentage',type=float,default=0.01)

parser.add_argument('--loss_mask',type=str,default = "1,1,1,1,1")

args = parser.parse_args()


if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True


train_dataset, test_dataset= scl_data.make_dataset(args)

train_loader = DataLoader(train_dataset, num_workers=2,batch_size=args.batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, num_workers=2,batch_size=args.eval_batch_size, shuffle=True, drop_last=True)

##make model
device = torch.device(args.gpu_ids)
config = RobertaConfig.from_pretrained("roberta-base")
config.num_labels = 2
pretrained_model = RobertaForSequenceClassification.from_pretrained("roberta-base",config=config)
model = scl_model(config,device,pretrained_model,args)


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

model = model.to(device)

step = 0
bar = tqdm.tqdm(total=args.steps)
bar.update(0)
best_acc = 0
recoder = Recoder(args)
loss_mask_str = args.loss_mask.split(',')
loss_mask = [float(i) for i in loss_mask_str]
print(loss_mask)

count = 0
begin_eval = False
while(step < args.steps):
    model.train()
    for batch in train_loader:
        # optimizer.zero_grad()
        ce_loss_x, ce_loss_s, ce_loss_semi, scl_loss, scl_loss_semi = model(batch)
        loss = loss_mask[0] * ce_loss_x + loss_mask[1] * ce_loss_s + loss_mask[2] * ce_loss_semi + loss_mask[3] * scl_loss + loss_mask[4] * scl_loss_semi

        loss.backward()

        count += 1
        if (count % args.num_accum == 0):
            optimizer.step()
            recoder.log_train(ce_loss_x, ce_loss_s, ce_loss_semi, scl_loss, scl_loss_semi,loss)
            step += 1
            optimizer.zero_grad()

            if (step % args.log_step == 0):
                begin_eval = True

            if (step % 10 == 0):
                bar.update(10)

        # step += 1
        
        
        if begin_eval:
            recoder.meter(step)
            evaluate_model(model,test_loader,recoder,step)
            torch.save(recoder, args.log_dir)
            
            # if (log["acc"] > best_acc):
                # best_acc = log["acc"]
                # torch.save(model.state_dict(),args.model_dir)
            model.train()
            begin_eval = False













