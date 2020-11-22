import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
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



parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='cnn')
parser.add_argument('--use_mlp', action='store_true')
parser.add_argument('--steps', type=int, default=10000)
parser.add_argument('--num_neg',type=int,default=16)
parser.add_argument('--summary_method',type=str,default="None")
parser.add_argument("--toy",action='store_true')
parser.add_argument("--local_rank",type=int, default=0)
parser.add_argument("--num_workers",type=int,default=1)
parser.add_argument("--toy_size",type=int,default=1000)
parser.add_argument("--seed",type=int,default=41)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len",type=int, default=256)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--K', type=int, default=65536)
parser.add_argument('--m', type=float, default=0.99)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay',type=float, default=1e-5)
parser.add_argument('--T', type=float, default=0.07)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clip',type=float,default=1)
parser.add_argument('--dist_func',type=str,default='cosin')
parser.add_argument('--n_vocab', type=int, default=30000)
parser.add_argument('--cnn_dim', type=int, default=256)
parser.add_argument('--dense1_dim', type=int, default=256)
parser.add_argument('--winsize', type=int, default=5)
parser.add_argument('--padding', type=int, default=1)

args = parser.parse_args()


if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if args.local_rank != -1:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend="nccl")


 ##make dataset
train_dataset = data.make_dataset(args)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, num_workers=2,batch_size=args.batch_size, shuffle=True, drop_last=True,  sampler=train_sampler)

##make model
base_model = RobertaModel.from_pretrained("roberta-base")
model = EmbNetwork(base_model, pooling_strategy='last').cuda()
model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank)

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

def batch_forward(model, batch):
    anchor_ids, pos_ids, neg_ids, neural_ids = [item.cuda() for item in batch]
    bsz = anchor_ids.shape[0]
    anchor_feature = model(anchor_ids).unsqueeze(1) #(bsz,1,768)
    hid_dim = anchor_feature.shape[2]

    pos_feature = model(pos_ids).unsqueeze(1)   #(bsz,1,768)

    neg_ids  = neg_ids.view(-1,hid_dim)
    neg_feature = model(neg_ids).view(bsz,-1,hid_dim)   #(bsz,num_neg, 768)

    neural_ids = neural_ids.view(-1, hid_dim)
    neural_feature = model(neural_ids).view(bsz,-1,hid_dim) #(bsz,num_neg, 768)

    return anchor_feature, pos_feature, neg_feature, neural_feature


def train(args, model,train_loader,optimizer):

    for iter_num in tqdm.tqdm(range(args.steps)):
        optimizer.zero_grad()
        batch = train_loader.next()
        anchor_feature, pos_feature, neg_feature, neural_feature = batch_forward(model,batch)

        d_pos = dist_function(anchor_feature, pos_feature,args.dist_func)
        d_neg = dist_function(anchor_feature, neg_feature,args.dist_func)
        d_neu = dist_function(anchor_feature, neural_feature,args.dist_func)

        loss_neg = triplet_loss(d_pos,d_neg,margin=2.0)
        loss_neu = triplet_loss(d_pos,d_neu,margin=1.0)

        loss = loss_neg + loss_neu

        loss.backward()
        
        optimizer.step()

        

        

def triplet_loss(d_pos,d_neg,margin=1.0,method='cosin',reduction='mean'):
    if method == 'cosin':
        loss = torch.clamp(d_neg - d_pos + margin, min=0.0)
    if method == 'ecludien':
        loss = None ##TO BE ADD

    if reduction=='mean':
        loss = torch.mean(loss)

    if reduction=='sum':
        loss = torch.sum(loss)

    return loss

def dist_function(x1,x2,method='cosin'):
    if method == 'cosin':
        dist = F.cosine_similarity(x1,x2,dim=2)
    
    if method == 'ecludien':
        dist = None         ##TO BE ADD

    return dist



# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# base_model = RobertaModel.from_pretrained("roberta-base")
# base_encoder = EmbNetwork(base_model, pooling_strategy='last')
# moco = MoCo(base_encoder, K=args.K, m=args.m, T=args.T, mlp=args.use_mlp)
# # train_loader = data.make_train_loader()
# train_loader = [(torch.LongTensor([[1,2,3,4]]),torch.LongTensor([[5,6,7,8]]))]
# optimizer = torch.optim.SGD(moco.encoder_q.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# criterion = nn.CrossEntropyLoss()


# for epoch in range(args.epochs):
#     for q_ids, k_ids in train_loader:

#         q_ids, k_ids = q_ids.to(device), k_ids.to(device)
#         # print(q_ids.shape)
#         # print(k_ids.shape)
#         logits, labels = moco.forward(q_ids, k_ids)
#         loss = criterion(logits, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()









