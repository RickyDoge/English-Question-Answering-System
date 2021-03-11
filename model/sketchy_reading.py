import torch
from model import baseline


class SketchyReadingModel(baseline.BaselineModel):
    def __init__(self, hidden_dim=256, clm_model='google/electra-small-discriminator'):
        super(SketchyReadingModel, self).__init__(hidden_dim, clm_model)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # batch_size * seq_length * hidden_dim
        last_hidden_state = self.pre_trained_clm(input_ids, attention_mask, token_type_ids).last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = torch.sigmoid(self.cls_fc_layer(cls_output))
        return cls_output