from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig
import torch.nn.functional as F
import torch.nn as nn
import copy
import torch
from supConloss import SupConLoss

class ClassificationHead(nn.Module):
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

class scl_model_multi(nn.Module):
    def __init__(self,config,device,pretrained_model,with_semi=True,with_sum=True):
        super().__init__()
        self.cls_x = ClassificationHead(config)
        self.cls_s = ClassificationHead(config)
        self.mlp_x = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))
        self.mlp_s = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))

        self.f = RobertaModel(config, add_pooling_layer=False)
        self.scl_criterion = SupConLoss(temperature=0.3,base_temperature = 0.3)
        self.ce_criterion = nn.CrossEntropyLoss()
        # self.f = copy.deepcopy(pretrained_enc)

        # self.f = RobertaModel(config)
        self.device = device
        self.init_weights(pretrained_model)
        self.with_semi = with_semi
        self.with_sum = with_sum
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

        ce_loss_x = self.ce_criterion(p_x,y_a)
        if self.with_semi:
            ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        else:
            ce_loss_s = self.ce_criterion(p_s,y_a)

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        if self.with_sum:
          z = torch.cat([z_x,z_s],dim=1)
        else:
          z = z_x

        if self.with_semi:
            scl_loss = (self.scl_criterion(z,labels = y_a) + self.scl_criterion(z,labels = y_b)) / 2
        else:
            scl_loss = self.scl_criterion(z,labels = y_a)

        

        return ce_loss_x, ce_loss_s, scl_loss



