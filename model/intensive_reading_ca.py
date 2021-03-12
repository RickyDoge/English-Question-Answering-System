import torch
import torch.nn as nn
from model import utils
from transformers import ElectraModel


class IntensiveReadingWithCrossAttention(nn.Module):
    def __init__(self, hidden_dim=256, clm_model='google/electra-small-discriminator'):
        super(IntensiveReadingWithCrossAttention, self).__init__()
        self.pre_trained_clm = ElectraModel.from_pretrained(clm_model)
        self.multi_head_attention = nn.MultiheadAttention

    def forward(self, input_ids, attention_mask, token_type_ids, sep_id):
        return 0

