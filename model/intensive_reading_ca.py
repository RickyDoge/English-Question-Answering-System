import math
import torch.nn as nn
from model import utils
from transformers import ElectraModel


class IntensiveReadingWithCrossAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1, clm_model='google/electra-small-discriminator'):
        super(IntensiveReadingWithCrossAttention, self).__init__()
        self.pre_trained_clm = ElectraModel.from_pretrained(clm_model)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout)
        self.span_detect_layer = nn.Linear(hidden_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.span_detect_layer.weight, gain=1 / math.sqrt(2))
        nn.init.constant_(self.span_detect_layer.bias, 0.)

    def forward(self, input_ids, attention_mask, token_type_ids, pad_idx, max_qus_length, max_con_length):
        # [batch_size * seq_length * hidden_dim]
        last_hidden_state = self.pre_trained_clm(input_ids, attention_mask, token_type_ids).last_hidden_state
        # question_hidden, passage_hidden: [batch_size * question/passage length * hidden_dim]
        # question_pad_mask, passage_pad_mask: [batch_size * question/passage length]
        question_hidden, passage_hidden, question_pad_mask, _ = \
            utils.generate_question_and_passage_hidden(last_hidden_state, attention_mask, token_type_ids, pad_idx,
                                                       max_qus_length, max_con_length)
        question_hidden = question_hidden.transpose(0, 1).to(last_hidden_state.device)
        passage_hidden = passage_hidden.transpose(0, 1).to(last_hidden_state.device)
        question_pad_mask = question_pad_mask.to(last_hidden_state.device)
        # https://stackoverflow.com/questions/65262928/query-padding-mask-and-key-padding-mask-in-transformer-encoder
        # It said we don't need to use query_padding_mask
        # attn_output: [passage_length * batch_size * hidden_dim]
        attn_output, _ = self.attention(passage_hidden, question_hidden, question_hidden,
                                        key_padding_mask=question_pad_mask,
                                        )
        # [batch_size * passage_length * 2]
        span_output = self.span_detect_layer(attn_output).transpose(0, 1)

        start_logits, end_logits = span_output.split(1, dim=-1)
        start_logits = start_logits.squeeze(dim=-1)
        end_logits = end_logits.squeeze(dim=-1)
        return start_logits, end_logits
