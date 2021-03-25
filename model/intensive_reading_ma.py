import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import utils
from model.classification_head import ElectraClassificationHead
from transformers import ElectraModel


class IntensiveReadingWithMatchAttention(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1, clm_model='google/electra-small-discriminator'):
        super(IntensiveReadingWithMatchAttention, self).__init__()
        self.pre_trained_clm = ElectraModel.from_pretrained(clm_model)
        self.cls_head = ElectraClassificationHead(hidden_dim=hidden_dim, dropout=dropout, num_labels=2)
        self.Hq_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.span_detect_layer = nn.Linear(hidden_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.Hq_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.span_detect_layer.weight, gain=1 / math.sqrt(2))
        nn.init.constant_(self.Hq_proj.bias, 0.)
        nn.init.constant_(self.span_detect_layer.bias, 0.)

    def forward(self, input_ids, attention_mask, token_type_ids, pad_idx, max_qus_length, max_con_length):
        # batch_size * seq_length * hidden_dim
        H = self.pre_trained_clm(input_ids, attention_mask, token_type_ids).last_hidden_state
        cls_output = torch.sigmoid(self.cls_head(H))
        question_hidden, passage_hidden, question_pad_mask, _ = \
            utils.generate_question_and_passage_hidden(H, attention_mask, token_type_ids, pad_idx, max_qus_length,
                                                       max_con_length)
        question_hidden = question_hidden.to(H.device)
        passage_hidden = passage_hidden.to(H.device)
        question_pad_mask = question_pad_mask.to(H.device)
       # question_hidden.masked_fill_(question_pad_mask.unsqueeze(dim=-1), value=float(-1e9))

        Hq = self.Hq_proj(question_hidden)
        # passage_hidden(Hp): [batch_size * context_length * hidden_dim], Hq: [batch_size * question_length * hidden_dim]
        attn_scores = torch.bmm(passage_hidden, Hq.transpose(1, 2))  # attn_scores: [batch_size * context_length * question_length]
        M = F.softmax(attn_scores, dim=-1)
        # M: [batch_size * context_length * question_length], Hq: [batch_size * question_length * hidden_dim]
        # Hp_pri: [batch_size * context_length * hidden_dim]
        Hp_pri = torch.bmm(M, Hq)

        # [batch_size * context_length * 2]
        span_output = self.span_detect_layer(Hp_pri)

        start_logits, end_logits = span_output.split(1, dim=-1)

        # [batch_size * context_length]
        start_logits = start_logits.squeeze(dim=-1)
        end_logits = end_logits.squeeze(dim=-1)
        return cls_output, start_logits, end_logits
