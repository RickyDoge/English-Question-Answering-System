import torch
import math


class DynamicWeightAveragingWrapper():
    # multi-task learning: dynamic weight averaging
    # End-to-End Multi-Task Learning with Attention，CVPR 2019
    # https://arxiv.org/abs/1803.10704
    def __init__(self, init_span_loss=6.0, init_cls_loss=3.5, scale=5.0, T=5.0):
        self.last_span_loss = init_span_loss
        self.last_cls_loss = init_cls_loss
        self.scale = scale
        self.T = T

    def loss(self, span_loss, cls_loss):
        with torch.no_grad():
            span_rate = span_loss.item() / self.last_span_loss / self.T
            span_rate = math.exp(span_rate)
            cls_rate = cls_loss.item() / self.last_cls_loss / self.T
            cls_rate = math.exp(cls_rate)
            sum = span_rate + cls_rate
        return (span_rate / sum) * span_loss + (cls_rate / sum) * self.scale * cls_loss
