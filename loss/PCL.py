import torch
from torch import nn
import torch.nn.functional as F

class PCLLoss(nn.Module):

    def __init__(self, verbalizer):
        super(PCLLoss, self).__init__()
        self.num_classes = verbalizer.num_classes

    def forward(self, embedding, features):
        loss = self.pcl_loss(embedding, features)
        return loss


    def pcl_loss(self, v_ins, t_ins):
        lossrank = 0.
        num = v_ins.shape[1]
        for i in range(v_ins.shape[0]):
            simlist = self.sim(v_ins, v_ins[i]).mean(dim=(-1,-2)).reshape(1, -1)
            target = self.sim(t_ins, t_ins[i]).mean(dim=(-1,-2)).reshape(1, -1)
            lossrank += self.list_mle(simlist, target)
        lossrank = lossrank / v_ins.shape[0]
        return lossrank
    

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1,0))
    
    @staticmethod
    def list_mle(y_pred, y_true, k=None):
        # y_pred : batch x n_items
        # y_true : batch x n_items 
        if k is not None:
            sublist_indices = (y_pred.shape[1] * torch.rand(size=k)).long()
            y_pred = y_pred[:, sublist_indices] 
            y_true = y_true[:, sublist_indices] 
    
        _, indices = y_true.sort(descending=True, dim=-1)
    
        pred_sorted_by_true = y_pred.gather(dim=1, index=indices)
    
        cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    
        listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true
    
        return listmle_loss.sum(dim=1).mean()
    
    