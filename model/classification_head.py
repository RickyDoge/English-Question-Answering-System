import torch.nn as nn
import torch.nn.functional as F


class ElectraClassificationHead(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1, num_labels=2):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, num_labels)

    def forward(self, last_layer_hidden):
        x = last_layer_hidden[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
