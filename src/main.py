import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import BertModel, BertConfig, BertForSequenceClassification
from transformers import XLNetModel, XLNetConfig, XLNetForSequenceClassification
from data import collate_fn, collate_fn_mix, multiLabelDataset, make_tokenizer,collate_fn_feature_mix
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
import torch.backends.cudnn as cudnn
import os
import tqdm
from opt import OpenAIAdam
import tqdm
from model import make_model

class Recoder_multi():
    def __init__(self,args):
        self.args = args
        self.ce_loss_x = []
        self.ce_loss_s = []
        self.scl_loss = []
        self.loss = []
        self.acc = []
        self.step = []
    def log_train(self,ce_loss_x, ce_loss_s, scl_loss, loss):
        self.ce_loss_x.append(ce_loss_x.item())
        self.ce_loss_s.append(ce_loss_s.item())
        self.scl_loss.append(scl_loss.item())
        self.loss.append(loss.item())
    
    def log_test(self,acc,step):
        self.acc.append(acc)
        self.step.append(step)


    def meter(self,step):
        st,ed = step - self.args.log_step, step
        print("===================================")
        print("step: ",step)
        print("loss: ",sum(self.loss[st:ed])/self.args.log_step)
        print("ce_loss_x: ",sum(self.ce_loss_x[st:ed])/self.args.log_step)
        print("ce_loss_s: ",sum(self.ce_loss_s[st:ed])/self.args.log_step)
        print("scl_loss: ",sum(self.scl_loss[st:ed])/self.args.log_step)
    



def evaluate_model(model, test_loader, recoder, step):
    print("Evaluation Start======")
    model.eval()

    # bar = tqdm.tqdm(total=len(test_loader))
    # bar.update(0)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x,x_perm,s,s_perm,y_a, y_b = batch
            x_ids = tokenizer(x, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
            # s_ids = tokenizer(s, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
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
parser.add_argument("--seed",type=int,default=41)
parser.add_argument("--gpu_ids",type=int,default=0)
parser.add_argument("--aug_methods",type=str,default='summary')
parser.add_argument("--with_mix", action='store_true')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--num_accum",type=int,default=1)
parser.add_argument("--max_len",type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip',type=float,default=1)
parser.add_argument("--lambd",type=float,default=0.8)
parser.add_argument('--log_step',type=int,default=100)
# parser.add_argument('--log_dir',type=str,default="finetune_log.pkl")
parser.add_argument("--model",type=str,default="xlnet")

parser.add_argument('--dataset',type=str,default="amazon")
parser.add_argument('--train_num',type=float,default=80)
parser.add_argument('--with_summary',action='store_true')
parser.add_argument("--feature_mix",action='store_true')

args = parser.parse_args()


if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

# Make Dataset
train_dataset = multiLabelDataset(dataset_name = args.dataset,max_num=args.train_num,seed=args.seed,split="train",aug_methods = args.aug_methods)
test_dataset = multiLabelDataset(dataset_name = args.dataset,max_num=1000,seed=args.seed,split="test",aug_methods = args.aug_methods)

if args.with_mix:
    my_collect = collate_fn_mix
else:
    my_collect = collate_fn

if args.feature_mix:
  my_collect = collate_fn_feature_mix

train_loader = DataLoader(train_dataset, num_workers=2, batch_size=args.batch_size, shuffle=True, collate_fn = my_collect)
test_loader = DataLoader(test_dataset, num_workers=2, batch_size=args.eval_batch_size,shuffle=False,collate_fn=my_collect)

# make tokenizer
tokenizer = make_tokenizer(args)


# make model
device = torch.device(args.gpu_ids)
model = make_model(args,device)
model = model.to(device)

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



step = 0
bar = tqdm.tqdm(total=args.steps)
bar.update(0)
best_acc = 0
recoder = Recoder_multi(args)

best_loss = float('inf')
count = 0
begin_eval = False

log_name = "Experiment/Seed-" + str(args.seed) + "/" + args.model + "/" + args.dataset + "/"
if args.with_mix:
  log_name += "with_mix_"
if args.with_summary:
  log_name += "with_summary_"
log_name += args.aug_methods
log_name += str(int(args.train_num)) + "_"
if args.feature_mix:
  log_name += "feature_mix_"
log_name += str(args.lambd) + ".pkl"


print("\n")
print(log_name)
while(step < args.steps):
    model.train()
    for x,x_perm,s,s_perm,y_a,y_b in train_loader:
        # print(batch)
        x_ids = tokenizer(x, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
        x_perm_ids = tokenizer(x_perm, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
        s_ids = tokenizer(s, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
        s_perm_ids = tokenizer(s_perm, padding = 'max_length', max_length = 200, truncation = True, return_tensors="pt")["input_ids"]
        # print(x_ids)
        batch = [x_ids,x_perm_ids,s_ids,s_perm_ids,y_a,y_b]
        if args.feature_mix:
          ce_loss_x, ce_loss_s, scl_loss = model.forward_feature_mix(batch)
        else:
          ce_loss_x, ce_loss_s, scl_loss = model(batch)
        ce_loss = (ce_loss_x + ce_loss_s)/2
        if not args.with_summary:
            ce_loss = ce_loss_x
        loss = args.lambd * ce_loss+ (1-args.lambd) * scl_loss

        # print(ce_loss_x, ce_loss_s, scl_loss)
        loss.backward()

        count += 1
        if (count % args.num_accum == 0):
            optimizer.step()
            recoder.log_train(ce_loss_x, ce_loss_s, scl_loss,loss)
            step += 1
            optimizer.zero_grad()

            if (step >= args.steps):
                break

            if (step % args.log_step == 0):
                begin_eval = True

            if (step % 10 == 0):
                bar.update(10)

        # step += 1
        
        
        if begin_eval:
            recoder.meter(step)
            evaluate_model(model,test_loader,recoder,step)
            torch.save(recoder, "../" + log_name)

            model.train()
            begin_eval = False













