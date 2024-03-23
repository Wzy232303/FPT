import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Optional
from Config import config
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
# from loss.Centerloss import CenterLoss
import math

class ProtoClVerbalizer(nn.Module):
    def __init__(self,
                args,
                lr: Optional[float] = 1e-3,
                epochs: Optional[int] = 5
                ):
        super(ProtoClVerbalizer, self).__init__()
        self.triplet = args.triplet
        # self.merge = False
        self.config = config(args)
        self.lr = lr
        self.epochs = epochs
        self.hidden_dims = 128
        self.num_classes = self.config.num_labels
        w = torch.empty((self.num_classes, self.hidden_dims))
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)
        self.head = nn.Linear(768, self.hidden_dims)
        self.optimizer = torch.optim.Adam(self.group_parameters, lr=self.lr)
        self.margin = args.margin
        self.scale = 0.05
        self.eps = 1e-7
        self.prompt_len = args.prompt_len
        # self.triplet_loss = TripletMarginWithSimDistanceLoss(self.margin)
        self.tripletloss = nn.TripletMarginWithDistanceLoss(
                                # distance_function=lambda x, y: (torch.log(1 - F.cosine_similarity(x, y))), 
                                # distance_function=lambda x, y: torch.exp(1- F.cosine_similarity(x, y)), 
                                distance_function=lambda x, y: 1- F.cosine_similarity(x, y), 
                                margin = self.margin,
                                reduction='none'
                            )
        self.soft_plus = nn.Softplus()
        

    @property
    def group_parameters(self,):
        r"""Include the last layer's parameters
        """
        if isinstance(self.head, torch.nn.Linear):
            return [p for n, p in self.head.named_parameters()] + [self.proto]
        else:
            return [p for n, p in self.head.named_parameters()] + [self.proto]
        
    @staticmethod
    def p2_dis(x, y):
        """
        x[batchsize, dim]
        y[numclass, dim]
        """
        bs = x.size(0)
        numclass = y.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(bs, numclass) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(numclass , bs).t()
        distmat.addmm_(1, -2, x.float(), y.t().float())
        return distmat

    def process_hiddens(self, hiddens: torch.Tensor, **kwargs):
        proto_logits = self.sim(hiddens, self.proto)
        # proto_logits = self.p2_dis(hiddens, self.proto)
        return proto_logits

    def process_outputs(self, outputs):## train_model or train_proto
        proto_logits = self.process_hiddens(self.head(outputs))
        return proto_logits
        # return self.ensemble_logits(soft_logits, proto_logits)

    def ensemble_logits(self, soft_logits, proto_logits):
        logits = torch.stack([soft_logits, proto_logits])
        logits = logits.permute(1,0,2)
        logits = self.scaler(logits)
        logits = torch.mean(logits, 1)
        return logits
        # return proto_logits



    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1,0))

    @staticmethod
    def sim_dot(x, y):
        return torch.matmul(x, y.transpose(1, 0))

    @staticmethod
    def scaler(logits):
        m = logits.mean(-1, keepdim=True)
        s = logits.std(-1, keepdim=True)
        return (logits - m) / s



    @staticmethod
    def minmax(x):
        max_ = x.max(-1, keepdim=True)[0]
        min_ = x.min(-1, keepdim=True)[0]
        return (x - min_) / (max_ - min_)



    def pcl_loss(self, v_ins):
        # instance-prototype loss
        sim_mat = torch.exp(self.sim(v_ins, self.proto)/self.scale)
        # print(sim_mat[0])
        num = sim_mat.shape[1]
        loss = 0.
        for i in range(num):
            pos_score = torch.diag(sim_mat[:,i,:])
            neg_score = (sim_mat[:,i,:].sum(1) - pos_score)
            loss += - torch.log(pos_score / (pos_score + neg_score)).sum()
        loss = loss / (num * self.num_classes * self.num_classes)
        # loss2 = 0.
        # sim_mat = sim_mat.permute(2,0,1)
        # for i in range(self.num_classes):
        #     pos_score = sim_mat[i,i,:]
        #     neg_score = (sim_mat[i:,:,:].sum(0) - pos_score)
        #     loss2 += - torch.log(pos_score / (pos_score + neg_score)).sum()
        # loss2 = loss2 / (num * self.num_classes * self.num_classes)


        # instance-instance loss

        loss_ins = 0.
        for i in range(v_ins.shape[0]):
            sim_instance = torch.exp(self.sim(v_ins, v_ins[i]) / self.scale)
            # print(sim_instance.shape)
            pos_ins = sim_instance[i]
            neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
            # print(pos_ins.shape, neg_ins.shape)
            loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
        loss_ins = loss_ins / (num * self.num_classes * num * self.num_classes)
        loss = loss + loss_ins
        return loss


    
    def triplet_loss(self, v_ins):
        # instance-prototype loss
        #v_ins.shape = num_classes, sample_num_per_class, dim
        num = v_ins.shape[1]
        dim = v_ins.shape[-1]
        loss = 0.
        loss1 = 0.
        for i in range(self.num_classes):
            ## select anchor: proto
            anchor = self.proto[i]
            for j in range(num):
                positive = v_ins[i][j]
                for k in range(self.num_classes):
                    if k == i:
                        continue
                    else:
                        negative = v_ins[k] # sample_num_per_class, dim
                        input1 = anchor.expand(num, dim)
                        input2 = positive.expand(num, dim)
                        input3 = negative
                        # loss1_list = self.triplet_loss(input1, input2, input3)
                        # focal_weights1 = (1 - torch.exp(-loss1_list)) ** 2
                        # loss1 += torch.mean(loss1_list * focal_weights1)
                        loss1 += torch.max(self.tripletloss(input1, input2, input3))
        loss1 = loss1 / (num * self.num_classes * (self.num_classes - 1))
        loss2 = 0.
        for i in range(self.num_classes):
            ## select anchor: instance
            for j in range(num):
                anchor = v_ins[i][j]
                positive = self.proto[i]
                for k in range(self.num_classes):
                    if k == i:
                        continue
                    else:
                        negative = self.proto[k] # 1, dim
                        input1 = anchor.unsqueeze(0)
                        input2 = positive.unsqueeze(0)
                        input3 = negative.unsqueeze(0)
                        loss2 += torch.max(self.tripletloss(input1, input2, input3))
                        # loss2_list = self.triplet_loss(input1, input2, input3)
                        # focal_weights2 = (1 - torch.exp(-loss2_list)) ** 2
                        # loss2 += torch.mean(loss2_list * focal_weights2)
        loss2 = loss2 / (num * self.num_classes * (self.num_classes - 1))
        loss = loss1 + loss2
        return 400 * loss
    
    def angle_triplet_center_loss(self, x):
        num = x.shape(1)
        centroids = self.proto
        centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
        intra_centroids = torch.cat(num * [centroids], -1)
        intra_centroids = intra_centroids.view(self.num_classes * num, self.hidden_dims)
        x = x.view(self.num_classes * num, self.hidden_dims)
        intra_d = torch.acos(torch.clamp(self.bdot(x, intra_centroids), -1.+self.eps, 1-self.eps))
        # dist_hinge = torch.clamp(intra_d - self.margin, min=0.0)
        # loss = torch.sum(dist_hinge)
        # return loss
        intra_d = intra_d.view(self.num_classes, num)
        intra_d, intra_idx = torch.max(intra_d, 1)
        idx = []
        for i in range(self.num_classes):
            idx.append(intra_idx[i] + i*num)
        idx = torch.stack(idx)
        maxd_x = x[idx]
        maxd_x = torch.cat((self.num_classes-1) * [maxd_x], -1)
        maxd_x = maxd_x.view(self.num_classes * (self.num_classes-1), self.hidden_dims)
        temp_centroids = torch.cat(self.num_classes * [centroids])
        inter_centroids = torch.tensor([]).to(self.device)
        n = self.num_classes+1
        for i in range(self.num_classes):
            inter_centroids = torch.cat([inter_centroids, temp_centroids[i*n+1:i*n+n]])
        inter_d = torch.acos(torch.clamp(self.bdot(maxd_x, inter_centroids), -1.+self.eps, 1-self.eps))
        inter_d = inter_d.view(self.num_classes, self.num_classes - 1)
        inter_d, inter_idx = torch.min(inter_d, 1)
        dist_hinge = torch.clamp(self.margin + intra_d - inter_d, min=0.0)
        loss = torch.sum(dist_hinge)
        return loss



    def train_proto(self, model, dataloader):
        model.eval()
        embeds = [[] for _ in range(self.num_classes)]
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input_ids, token_type_ids, attention_mask, labels, c, w, s, p, mask_token_index  = data
                _, outputs_at_mask = model((input_ids, token_type_ids, attention_mask), c, w, s, p, mask_token_index)
                assert outputs_at_mask.shape[0] == labels.shape[0]
                for j in range(len(outputs_at_mask)):
                    label = labels[j].item()
                    embeds[label].append(outputs_at_mask[j])
        embeds = [torch.stack(e) for e in embeds]
        embeds = torch.stack(embeds)
        loss = 0.
        for epoch in range(self.epochs):
            with autocast():
                x = self.head(embeds)
                if self.triplet:
                    loss = self.triplet_loss(x)
                    # print(loss.item())
                else:
                    loss = self.pcl_loss(x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.trained = True