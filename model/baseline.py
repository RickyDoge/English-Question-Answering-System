import torch
import torch.nn as nn
import torch.nn.functional as F
from model import utils
import math
from transformers import ElectraModel

class BaselineModel(nn.Module):
    def __init__(self, hidden_dim=256, clm_model='google/electra-small-discriminator'):
        super(BaselineModel, self).__init__()
        self.pre_trained_clm = ElectraModel.from_pretrained(clm_model)
        self.cls_fc_layer = nn.Linear(hidden_dim, 2, bias=True)
        self.span_detect_layer = nn.Linear(hidden_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.cls_fc_layer.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.span_detect_layer.weight, gain=1 / math.sqrt(2))
        nn.init.constant_(self.cls_fc_layer.bias, 0.)
        nn.init.constant_(self.span_detect_layer.bias, 0.)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # batch_size * seq_length * hidden_dim
        last_hidden_state = self.pre_trained_clm(input_ids, attention_mask, token_type_ids).last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = torch.sigmoid(self.cls_fc_layer(cls_output))
        # batch_size * seq_length * 2
        span_output = self.span_detect_layer(last_hidden_state)

        # start span logit vector and end span logit vector
        start_logits, end_logits = span_output.split(1, dim=-1)
        start_logits = start_logits.squeeze(dim=-1)
        end_logits = end_logits.squeeze(dim=-1)
        return cls_output, start_logits, end_logits
