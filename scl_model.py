from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig
import torch.nn.functional as F
import torch.nn as nn
import copy
import torch
from supConloss import SupConLoss

criterion = nn.CrossEntropyLoss()
scl_criterion = SupConLoss(temperature=0.3,base_temperature = 0.3)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class scl_model(nn.Module):
    def __init__(self,config,device,pretrained_model,args):
        super().__init__()
        self.cls_x = RobertaClassificationHead(config=config)
        self.cls_s = RobertaClassificationHead(config=config)

        self.f = RobertaModel(config, add_pooling_layer=False)

        # self.f = RobertaModel(config)
        self.device = device
        self.init_weights(pretrained_model)
        self.t_pos = args.t_pos
        self.t_mix = args.t_mix
        self.trade_off = args.trade_off
    def init_weights(self,pretrained_model):
        self.cls_x = copy.deepcopy(pretrained_model.classifier)
        self.cls_s = copy.deepcopy(pretrained_model.classifier)
        self.f = copy.deepcopy(pretrained_model.roberta)

    def predict(self,x):
        f_x = self.f(x)[0]
        score = self.cls_x(f_x)
        return score
    def forward(self,batch):
        x_pos,x_neg,s_pos,s_neg,s_semi = batch
        batch_size = x_pos.shape[0]
        x = torch.cat([x_pos,x_neg],dim=0).to(self.device)
        s = torch.cat([s_pos,s_neg],dim=0).to(self.device)
        y = torch.cat([torch.ones(x_pos.shape[0]),torch.zeros(x_neg.shape[0])],dim=0).long().to(self.device)
        f_x = self.f(x)[0]
        f_s = self.f(s)[0]
        f_s_semi = self.f(s_semi.to(self.device))[0]

        p_x = self.cls_x(f_x)
        p_s = self.cls_s(f_s)
        p_s_semi = self.cls_s(f_s_semi)
        ce_loss_x = criterion(p_x, y)
        ce_loss_s = criterion(p_s, y)
        ce_loss_semi = -0.5 * (F.log_softmax(p_s_semi).sum() / p_s_semi.shape[0])

        f_x, f_s, f_s_semi = f_x[:,0,:], f_s[:,0,:], f_s_semi[:,0,:]

        l_pos = torch.bmm(f_x[:batch_size].unsqueeze(1), f_s[:batch_size].unsqueeze(2)).squeeze(1) #B * 1
        neg_f = torch.cat([f_x[batch_size:],f_s[batch_size:]],dim=0).to(self.device) # 2B * 768
        l_neg = torch.mm(f_x[:batch_size],neg_f.transpose(0,1)) # B * 2B
        logits = torch.cat([l_pos,l_neg],dim = 1).to(self.device)
        labels = torch.zeros(batch_size).long().to(self.device)
        scl_loss = criterion(logits/self.t_pos, labels)

        l_semi_pos = torch.bmm(f_x[:batch_size].unsqueeze(1), f_s_semi.unsqueeze(2)).squeeze(1) #B * 1
        l_semi_neg = torch.bmm(f_x[batch_size:].unsqueeze(1), f_s_semi.unsqueeze(2)).squeeze(1) #B * 1
        comp_f = torch.cat([f_x, f_s],dim=0).to(self.device) #4B * C
        l_comp = torch.mm(f_s_semi,comp_f.transpose(0,1)) # B * 4B
        logits = torch.cat([l_semi_pos,l_comp],dim = 1).to(self.device) # B * (1 + 4B)
        scl_loss_semi = self.trade_off * criterion(logits/self.t_mix, labels)

        logits[:,0] = l_semi_neg.squeeze(1)
        scl_loss_semi += (1- self.trade_off) * criterion(logits/self.t_mix, labels)

        return ce_loss_x, ce_loss_s, ce_loss_semi, scl_loss, scl_loss_semi

class scl_model_multi(nn.Module):
    def __init__(self,config,device,pretrained_model,with_semi=True):
        super().__init__()
        self.cls_x = RobertaClassificationHead(config=config)
        self.cls_s = RobertaClassificationHead(config=config)
        self.mlp_x = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))
        self.mlp_s = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))

        self.f = RobertaModel(config, add_pooling_layer=False)

        # self.f = RobertaModel(config)
        self.device = device
        self.init_weights(pretrained_model)
        self.with_semi = with_semi
    def init_weights(self,pretrained_model):
        self.cls_x = copy.deepcopy(pretrained_model.classifier)
        self.cls_s = copy.deepcopy(pretrained_model.classifier)
        self.f = copy.deepcopy(pretrained_model.roberta)
        for p in self.mlp_x.parameters():
            # if p.dim() > 2:
                # torch.nn.init.xavier_normal_(p)
            torch.nn.init.normal_(p)

        for p in self.mlp_s.parameters():
            # if p.dim() > 2:
                # torch.nn.init.xavier_normal_(p)
            torch.nn.init.normal_(p)

    def predict(self,x):
        f_x = self.f(x)[0]
        score = self.cls_x(f_x)
        return score
    def forward(self,batch):
        x,s_mix,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = self.f(x)[0]
        f_s = self.f(s_mix)[0]

        p_x = self.cls_x(f_x)
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s)
        
        # print(p_x.shape)
        # print(y_a.shape)

        ce_loss_x = criterion(p_x,y_a)
        if self.with_semi:
            ce_loss_s = (criterion(p_s,y_a) + criterion(p_s,y_b)) / 2
        else:
            ce_loss_s = criterion(p_s,y_a)

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        
        z = torch.cat([z_x,z_s],dim=1)

        if self.with_semi:
            scl_loss = (scl_criterion(z,labels = y_a) + scl_criterion(z,labels = y_b)) / 2
        else:
            scl_loss = scl_criterion(z,labels = y_a)

        ucl_loss = scl_criterion(z)

        return ce_loss_x, ce_loss_s, scl_loss, ucl_loss



