import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from Config import config

class SoftVerbalizer(nn.Module):
    def __init__(self,
                args,
                ):
        super(SoftVerbalizer, self).__init__()
        self.config = config(args)
        self.hidden_dims = 768
        self.num_classes = self.config.num_labels
        self.soft = torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        proto_logits = self.soft(hiddens)
        return proto_logits

    def process_outputs(self, outputs):## train_model or train_proto
        proto_logits = self.process_hiddens(outputs) ## 处理mask hidden
        return proto_logits
    

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1,0))