import math
import torch
import torch.nn as nn
from model import utils
from model.classification_head import ElectraClassificationHead
from transformers import ElectraModel


class IntensiveReadingWithConvolutionNet(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1, out_channel=8, filter_size=3,
                 clm_model='google/electra-small-discriminator'):
        super(IntensiveReadingWithConvolutionNet, self).__init__()
        self.pre_trained_clm = ElectraModel.from_pretrained(clm_model)
        self.cls_head = ElectraClassificationHead(hidden_dim=hidden_dim, dropout=dropout, num_labels=2)
        self.conv = TextConvolutionClassificationHead(out_channel=out_channel, filter_size=filter_size,
                                                      hidden_dim=hidden_dim)

    def forward(self, input_ids, attention_mask, token_type_ids, pad_idx, max_qus_length, max_con_length):
        # [batch_size * seq_length * hidden_dim]
        last_hidden_state = self.pre_trained_clm(input_ids, attention_mask, token_type_ids).last_hidden_state
        cls_output = torch.sigmoid(self.cls_head(last_hidden_state))

        # question_hidden, passage_hidden: [batch_size * question/passage length * hidden_dim]
        # question_pad_mask, passage_pad_mask: [batch_size * question/passage length]
        _, passage_hidden, _, _ = \
            utils.generate_question_and_passage_hidden(last_hidden_state, attention_mask, token_type_ids, pad_idx,
                                                       max_qus_length, max_con_length)
        passage_hidden = passage_hidden.to(last_hidden_state.device)

        # [batch_size * passage_length * 2]
        span_output = self.conv(passage_hidden)
        start_logits, end_logits = span_output.split(1, dim=-1)
        start_logits = start_logits.squeeze(dim=-1)
        end_logits = end_logits.squeeze(dim=-1)
        return cls_output, start_logits, end_logits


class TextConvolutionClassificationHead(nn.Module):
    def __init__(self, out_channel=8, filter_size=3, hidden_dim=256, dropout=0.5):
        super(TextConvolutionClassificationHead, self).__init__()
        assert filter_size == 3 or filter_size == 5, 'filter_size only allows to be 3 and 5'
        pad_length = 1 if filter_size == 3 else 2
        self.conv = nn.Conv2d(1, out_channel, kernel_size=(filter_size, hidden_dim), padding=(pad_length, 0))
        self.dropout = nn.Dropout(dropout)
        self.cls_layer = nn.Linear(out_channel, 2)
        nn.init.xavier_uniform_(self.conv.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.cls_layer.weight, gain=1 / math.sqrt(2))

    def forward(self, passage_hidden):
        # input: [batch_size * passage_length * hidden_dim]
        x = self.conv(passage_hidden.unsqueeze(dim=1))
        x = torch.relu(self.dropout(x))
        x = x.squeeze(dim=-1).transpose(1, 2)  #: [batch_size * passage_length * num_channels]
        x = self.cls_layer(x)
        return x
