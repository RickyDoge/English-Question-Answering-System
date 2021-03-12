import torch
import torch.nn as nn
import math
from transformers import ElectraModel


class SketchyReadingModel(nn.Module):
    def __init__(self, hidden_dim=256, clm_model='google/electra-small-discriminator'):
        super(SketchyReadingModel, self).__init__()
        self.pre_trained_clm = ElectraModel.from_pretrained(clm_model)
        self.cls_fc_layer = nn.Linear(hidden_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.cls_fc_layer.weight, gain=1 / math.sqrt(2))
        nn.init.constant_(self.cls_fc_layer.bias, 0.)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # batch_size * seq_length * hidden_dim
        last_hidden_state = self.pre_trained_clm(input_ids, attention_mask, token_type_ids).last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = torch.sigmoid(self.cls_fc_layer(cls_output))
        return cls_output