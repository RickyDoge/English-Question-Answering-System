import math
import torch
import torch.nn as nn
from model.classification_head import ElectraClassificationHead
from model.intensive_reading_cnn import IntensiveReadingWithConvolutionNet
from transformers import ElectraModel


class IntensiveReadingWithConvolutionNetAndSkip(IntensiveReadingWithConvolutionNet):
    def __init__(self, hidden_dim=256, dropout=0.1, out_channel=8, filter_size=3,
                 clm_model='google/electra-small-discriminator'):
        super(IntensiveReadingWithConvolutionNetAndSkip, self).__init__()
        self.pre_trained_clm = ElectraModel.from_pretrained(clm_model)
        self.cls_head = ElectraClassificationHead(hidden_dim=hidden_dim, dropout=dropout, num_labels=2)
        self.conv = TextConvolutionClassificationHeadWithSkipConnection(out_channel=out_channel,
                                                                        filter_size=filter_size,
                                                                        hidden_dim=hidden_dim)


class TextConvolutionClassificationHeadWithSkipConnection(nn.Module):
    def __init__(self, out_channel=8, filter_size=3, hidden_dim=256):
        super(TextConvolutionClassificationHeadWithSkipConnection, self).__init__()
        assert filter_size == 3 or filter_size == 5, 'filter_size only allows to be 3 and 5'
        pad_length = 1 if filter_size == 3 else 2
        self.conv = nn.Conv2d(1, out_channel, kernel_size=(filter_size, hidden_dim), padding=(pad_length, 0))
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.start_layer = nn.Linear(out_channel, 1)
        self.end_layer = nn.Linear(out_channel * 2, 1)
        nn.init.xavier_uniform_(self.conv.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.start_layer.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.end_layer.weight, gain=1 / math.sqrt(2))

    def forward(self, passage_hidden):
        # input: [batch_size * passage_length * hidden_dim]
        x = self.conv(passage_hidden.unsqueeze(dim=1))
        x = torch.relu(self.batchNorm(x))
        x = x.squeeze(dim=-1).transpose(1, 2)  # [batch_size * passage_length * num_channels]
        start = self.start_layer(x)  # [batch_size * passage_length * 1]
        start_index = start.argmax(dim=1)  # [batch_size * 1 * 1]
        end = self.end_layer(torch.cat([x[start_index], x], dim=-1))  # [batch_size * passage_length * 1]
        return torch.cat([start, end], dim=-1)
