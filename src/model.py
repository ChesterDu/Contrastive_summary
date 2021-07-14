from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import BertModel, BertConfig, BertForSequenceClassification
from transformers import XLNetModel, XLNetConfig, XLNetForSequenceClassification
import torch.nn.functional as F
import torch.nn as nn
import copy
import torch
from supConloss import SupConLoss


def make_model(args,device):
  if args.model == "roberta":
    config = RobertaConfig.from_pretrained("roberta-base")
    config.num_labels = 5
    if args.dataset == "imdb":
      config.num_labels = 2
    if args.dataset == "ag_news":
      config.num_labels = 4
    if args.dataset == "yahoo":
      config.num_labels = 10
    pretrained_model = RobertaForSequenceClassification.from_pretrained("roberta-base",config=config)
    return scl_model_Roberta(config,device,pretrained_model,with_semi=args.with_mix,with_sum=args.with_summary)

  if args.model == "bert":
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_labels = 5
    if args.dataset == "imdb":
      config.num_labels = 2
    if args.dataset == "ag_news":
      config.num_labels = 4
    if args.dataset == "yahoo":
      config.num_labels = 10
    pretrained_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",config=config)
    return scl_model_Bert(config,device,pretrained_model,with_semi=args.with_mix,with_sum=args.with_summary)

  if args.model == "xlnet":
    config = XLNetConfig.from_pretrained("xlnet-base-cased")
    config.num_labels = 5
    if args.dataset == "imdb":
      config.num_labels = 2
    if args.dataset == "ag_news":
      config.num_labels = 4
    if args.dataset == "yahoo":
      config.num_labels = 10
    pretrained_model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",config=config)
    return scl_model_Xlnet(config,device,pretrained_model,with_semi=args.with_mix,with_sum=args.with_summary)


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

class scl_model_Roberta(nn.Module):
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
    def forward_feature_mix(self,batch):

        x,x_perm,s,s_perm,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = 0.5 * self.f(x)[0] + 0.5 * self.f(x_perm)[0]
        f_s = 0.5 * self.f(s)[0] + 0.5 * self.f(s_perm)[0]

        p_x = self.cls_x(f_x)
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s)
        
        # print(p_x.shape)
        # print(y_a.shape)

        
        ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        ce_loss_x = (self.ce_criterion(p_x,y_a) + self.ce_criterion(p_x,y_b)) / 2

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        z = torch.cat([z_x,z_s],dim=1)

        scl_loss = (self.scl_criterion(z,labels = y_a) + self.scl_criterion(z,labels = y_b)) / 2

        

        return ce_loss_x, ce_loss_s, scl_loss



class scl_model_Xlnet(nn.Module):
    def __init__(self,config,device,pretrained_model,with_semi=True,with_sum=True):
        super().__init__()
        self.cls_x = nn.Linear(config.d_model, config.num_labels)
        self.cls_s = nn.Linear(config.d_model, config.num_labels)
        self.mlp_x = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))
        self.mlp_s = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))

        self.f = XLNetModel(config)
        self.scl_criterion = SupConLoss(temperature=0.3,base_temperature = 0.3)
        self.ce_criterion = nn.CrossEntropyLoss()
        # self.f = copy.deepcopy(pretrained_enc)

        # self.f = RobertaModel(config)
        self.device = device
        self.init_weights(pretrained_model)
        self.with_semi = with_semi
        self.with_sum = with_sum
    def init_weights(self,pretrained_model):
        self.cls_x = copy.deepcopy(pretrained_model.logits_proj)
        self.cls_s = copy.deepcopy(pretrained_model.logits_proj)
        self.f = copy.deepcopy(pretrained_model.transformer)
          
        for p in self.mlp_x.parameters():
            torch.nn.init.normal_(p)

        for p in self.mlp_s.parameters():
            torch.nn.init.normal_(p)

    def predict(self,x):
        f_x = self.f(x)[0]
        score = self.cls_x(f_x[:,0,:])
        return score
    def forward(self,batch):
        x,s_mix,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = self.f(x)[0]
        f_s = self.f(s_mix)[0]

        p_x = self.cls_x(f_x[:,0,:])
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s[:,0,:])
        
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


class scl_model_Bert(nn.Module):
    def __init__(self,config,device,pretrained_model,with_semi=True,with_sum=True):
        super().__init__()
        self.cls_x = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_s = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout_x = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_s = nn.Dropout(config.hidden_dropout_prob)

        self.mlp_x = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))
        self.mlp_s = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))

        self.f = BertModel(config)
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
        self.f = copy.deepcopy(pretrained_model.bert)
          
        for p in self.mlp_x.parameters():
            torch.nn.init.normal_(p)

        for p in self.mlp_s.parameters():
            torch.nn.init.normal_(p)

    def predict(self,x):
        f_x = self.f(x)[0]
        score = self.cls_x(f_x[:,0,:])
        return score
    def forward(self,batch):
        x,s_mix,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = self.f(x)[0]
        f_s = self.f(s_mix)[0]

        p_x = self.cls_x(f_x[:,0,:])
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s[:,0,:])
        
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

        

        return ce_loss_x, ce_loss_s, scl_loss



