import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
from scl_multi_data import collate_fn, collate_fn_mixs, multiLabelDataset
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
import torch.backends.cudnn as cudnn
from builder import MoCo
import data
import scl_data
import os
import tqdm
from opt import OpenAIAdam
from train import train
import tqdm
from scl_model import scl_model,scl_model_multi


class Recoder_multi():
    def __init__(self,args):
        self.args = args
        self.ce_loss_x = []
        self.ce_loss_s = []
        self.scl_loss = []
        self.ucl_loss = []
        self.loss = []
        self.acc = []
        self.step = []
    def log_train(self,ce_loss_x, ce_loss_s, scl_loss, ucl_loss, loss):
        self.ce_loss_x.append(ce_loss_x.item())
        self.ce_loss_s.append(ce_loss_s.item())
        self.scl_loss.append(scl_loss.item())
        self.ucl_loss.append(ucl_loss.item())
        self.loss.append(loss.item())
    
    def log_test(self,acc,step):
        self.acc.append(acc)
        self.step.append(step)


    def meter(self,step):
        print("===================================")
        print("step: ",step)
        print("loss: ",sum(self.loss)/step)
        print("ce_loss_x: ",sum(self.ce_loss_x)/step)
        print("ce_loss_s: ",sum(self.ce_loss_s)/step)
        print("scl_loss: ",sum(self.scl_loss)/step)
        print("ucl_loss: ",sum(self.ucl_loss)/step)
    



def evaluate_model(model, test_loader, recoder, step):
    print("Evaluation Start======")
    model.eval()

    bar = tqdm.tqdm(total=len(test_loader))
    bar.update(0)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x_ids, s_mix_ids, y_a, y_b = batch
            seq_ids = x_ids.to(device)
            labels = y_a.to(device)
            logits = model.predict(seq_ids)

            prediction = torch.argmax(logits, dim = 1)
            correct += (prediction == labels).sum().item()
            total += prediction.shape[0]

    acc = correct / total
    print("Acc: ",acc)

    recoder.log_test(acc,step)




parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=10000)
parser.add_argument("--load_pretrain",action='store_true')
parser.add_argument("--seed",type=int,default=41)
parser.add_argument("--gpu_ids",type=int,default=0)
parser.add_argument("--with_mix", action='store_true')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--num_accum",type=int,default=1)
parser.add_argument("--max_len",type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip',type=float,default=1)
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

if args.dataset_pth != "none":
    raw_data = torch.load(args.dataset_pth)
    raw_train_data = raw_data["train"]
    raw_test_data = raw_data["test"]

random.shuffle(raw_train_data)
train_num = int(args.percentage * len(raw_train_data))
print("train num:",train_num)
train_dataset = multiLabelDataset(raw_train_data[:train_num])
test_dataset = multiLabelDataset(raw_test_data)

if args.with_mix:
    my_collect = collate_fn_mixs
else:
    my_collect = collate_fn
train_loader = DataLoader(train_dataset, num_workers=2, batch_size=args.batch_size, shuffle=True, collate_fn = my_collect)
test_loader = DataLoader(test_dataset, num_workers=2, batch_size=args.eval_batch_size,shuffle=False,collate_fn=my_collect)

# ##make model
device = torch.device(args.gpu_ids)
config = RobertaConfig.from_pretrained("roberta-base")
config.num_labels = 5
pretrained_model = RobertaForSequenceClassification.from_pretrained("roberta-base",config=config)
if args.load_pretrain:
    model = scl_model_multi(config,device,pretrained_model,with_semi=args.with_mix)
    model.load_state_dict(torch.load(args.model_dir,map_location="cpu"))
else:
    model = scl_model_multi(config,device,pretrained_model,with_semi=args.with_mix)

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
        # optimizer.zero_grad()
        # print(batch)
        ce_loss_x, ce_loss_s, scl_loss, ucl_loss = model(batch)
        loss = loss_mask[0] * ce_loss_x + loss_mask[1] * ce_loss_s + loss_mask[2] * scl_loss + loss_mask[3] * ucl_loss

        # print(ce_loss_x, ce_loss_s, scl_loss)
        loss.backward()

        count += 1
        if (count % args.num_accum == 0):
            optimizer.step()
            recoder.log_train(ce_loss_x, ce_loss_s, scl_loss,ucl_loss,loss)
            step += 1
            optimizer.zero_grad()

            if (step > args.steps):
                break

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
            #     best_acc = log["acc"]
            #     torch.save(model.state_dict(),args.model_dir)
            # if sum(recoder.loss) / step < best_loss:
            #     torch.save(model.state_dict(),args.model_dir)
            #     best_loss = sum(recoder.loss) / step

            model.train()
            begin_eval = False













