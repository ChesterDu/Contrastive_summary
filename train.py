import tqdm
import torch
import torch.nn.functional as F

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
