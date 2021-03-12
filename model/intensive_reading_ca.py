import torch
import torch.nn as nn
from model import utils
from transformers import ElectraModel


class IntensiveReadingWithCrossAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1, clm_model='google/electra-small-discriminator'):
        super(IntensiveReadingWithCrossAttention, self).__init__()
        self.pre_trained_clm = ElectraModel.from_pretrained(clm_model)
        self.multi_head_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, pad_idx):
        # batch_size * seq_length * hidden_dim
        last_hidden_state = self.pre_trained_clm(input_ids, attention_mask, token_type_ids).last_hidden_state
        utils.generate_question_and_passage_hidden(last_hidden_state, attention_mask, token_type_ids, pad_idx)

