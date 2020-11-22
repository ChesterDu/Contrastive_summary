import torch
import torch.nn as nn
import transformers
# import sentence_transformers
import copy


class EmbNetwork(nn.Module):
    def __init__(self, base_model, pooling_strategy="avg", use_mlp = False, out_dim = 256):
        super(EmbNetwork, self).__init__()

        self.base_model = copy.deepcopy(base_model)
        self.pooling = pooling_strategy
        

    def forward(self, x, attention_mask=None, pad_id=1):
        # input X of shape (bsz, seq_len)

        if self.pooling == "avg":
            if attention_mask == None:
                attention_mask = (x != pad_id)

            token_features = self.base_model(x)[0]  # (bsz,len,1024)
            token_features = token_features * attention_mask.unsqueeze(-1).float()
            sen_feature = token_features.sum(1)
            sen_len = attention_mask.sum(1)
            sen_feature = sen_feature / sen_len.unsqueeze(-1)

        if self.pooling == "last":
            sen_feature = self.base_model(x)[1]

        return sen_feature





