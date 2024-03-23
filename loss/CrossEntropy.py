import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.)

    def forward(self, logits, label):
        return self.loss(logits, label)