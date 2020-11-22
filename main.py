import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
import argparse
from models import EmbNetwork
from builder import MoCo
import data



parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='cnn')
parser.add_argument('--use_mlp', action='store_true')
parser.add_argument('--steps', type=int, default=10000)
parser.add_argument('--num_neg',type=int,default=16)
parser.add_argument('--summary_method',type=str,default="None")
parser.add_argument("--toy",action='store_true')
parser.add_argument("--toy_size",type=int,default=1000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--K', type=int, default=65536)
parser.add_argument('--m', type=float, default=0.99)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay',type=float, default=1e-5)
parser.add_argument('--T', type=float, default=0.07)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_vocab', type=int, default=30000)
parser.add_argument('--cnn_dim', type=int, default=256)
parser.add_argument('--dense1_dim', type=int, default=256)
parser.add_argument('--winsize', type=int, default=5)
parser.add_argument('--padding', type=int, default=1)

args = parser.parse_args()


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

train_loader = data.make_dataloader(args)
for batch in train_loader:
    for item in batch:
        print(item.shape)
    break







