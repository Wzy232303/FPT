import torch
import torch.nn.functional as F
from tqdm import tqdm
from runer.Baserunner import Baserunner
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast as autocast
from Config import config
import sklearn.metrics as sm
from loss.CrossEntropy import CrossEntropyLoss
from loss.PCL import PCLLoss

class RArunner(Baserunner):
    def __init__(self, model, trainiter, deviter, testiter, args) -> None:
        super().__init__(model, trainiter, deviter, testiter, args)
        self.calibration = args.calibration
        self.do_fewshot = args.do_fewshot
        self.celoss = CrossEntropyLoss()
        self.pcl = PCLLoss(self.model.module.verbalizer)
        self.modelname = args.model_name

    def train_epoch(self, optimizer, scheduler, epoch, is_fewshot=True):
        self.model.train()
        loss = 0.
        with tqdm(total=len(self.trainiter), desc='Epoch {}'.format(epoch + 1), ncols=100) as pbar:
            for batch in self.trainiter:
                input_ids, token_type_ids, attention_mask, label = batch[0:4]
                others = batch[4:]
                # with autocast():
                input = (input_ids, token_type_ids, attention_mask)
                outputs = self.model(input, *others)
                logits = outputs[0]
                # print(logits.tolist())
                CEloss = self.celoss(logits, label)
                loss = CEloss
                optimizer[0].zero_grad()
                if self.modelname == 'bert_ptuing':
                    optimizer[1].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                optimizer[0].step()
                scheduler[0].step()
                if self.modelname == 'bert_ptuing':
                    optimizer[1].step()
                    scheduler[1].step()

        if self.calibration:
            for _ in range(5):
                embeds = [[] for _ in range(self.config.num_labels)]
                featrs = [[] for _ in range(self.config.num_labels)]
                for batch in self.trainiter:
                    label = batch[3]
                    c, w, s, p = batch[4:8]
                    features = torch.cat((c, w, s, p), dim=-1)
                    embedding = torch.cat([l(features).unsqueeze(1) for l in self.model.module.prefix_transes], dim=1)
                    embedding = nn.functional.avg_pool2d(embedding, kernel_size=(embedding.shape[1], 1), stride=1).squeeze(1)
                    for j in range(len(embedding)):
                        label_ = label[j].item()
                        embeds[label_].append(embedding[j])
                        featrs[label_].append(features[j])
                embeds = [torch.stack(e) for e in embeds]
                embeds = torch.stack(embeds)
                featrs = [torch.stack(f) for f in featrs]
                featrs = torch.stack(featrs)
                pclloss = self.pcl(embeds, featrs)
                # print(pclloss.item())
                optimizer[-1].zero_grad()
                pclloss.backward()
                optimizer[-1].step()

    def get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        if self.modelname == 'bert_ptuing':
            print(1)
            exclude_layers = ['lstm_head', 'mlp_head', 'prefix_encoder']
            param_optimizer = [(name, param) for name, param in param_optimizer if not any(exclude_layer in name for exclude_layer in exclude_layers)]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_training_steps = len(self.trainiter) * self.config.num_epochs
        optimizer1 = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, correct_bias=False)
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer1, num_warmup_steps=int(num_training_steps * self.config.warm_up_ratio), num_training_steps=num_training_steps
        )
        optlist = [optimizer1]
        schelist = [scheduler1]
        if self.modelname == 'bert_ptuing':
            embedding_parameters = [
                    {'params': [p for p in self.model.module.lstm_head.parameters()]},
                    {'params': [p for p in self.model.module.mlp_head.parameters()]},
                    {'params': [p for p in self.model.module.prefix_encoder.parameters()]}
            ]
            optimizer2 = AdamW(embedding_parameters, lr=5e-5, correct_bias=False)
            optlist.append(optimizer2)
            scheduler2 = get_linear_schedule_with_warmup(
                optimizer2, num_warmup_steps=int(num_training_steps * self.config.warm_up_ratio), num_training_steps=num_training_steps
            )
            schelist.append(scheduler2)
        if self.calibration:
            optimizer3 = torch.optim.Adam([p for n, p in self.model.module.prefix_transes.named_parameters()], lr=1e-3)
            optlist.append(optimizer3)
        return optlist, schelist

    def train(self):
        if self.config.verb == 'proto':
            self.model.module.verbalizer.train_proto(self.model, self.trainiter)
        super().train()


    def evaluate(self, iter):
        correct = 0
        total = 0
        pred_list = []
        label_list = []
        with torch.no_grad():
            running_loss = 0
            for batch in iter:
                input_ids, token_type_ids, attention_mask, label = batch[0:4]
                others = batch[4:]
                # with autocast():
                input = (input_ids, token_type_ids, attention_mask)
                outputs = self.model(input, *others)
                logits = outputs[0]
                loss = self.config.loss_func(logits, label)
                pred = torch.max(logits, dim=1)[1]
                pred_list.extend(pred.cpu().numpy().tolist())
                label_list.extend(label.cpu().numpy().tolist())
                correct += (pred == label).sum().item()
                total += label.size(0)
                running_loss += loss.item()
            # if self.do_cl:
            # # print(pred_list)
            # # print(label_list)
            #   assert len(pred_list) == len(label_list)
                # cm = sm.confusion_matrix(label_list, pred_list)
                # print("---------------混淆矩阵\n", cm)

            # cp = sm.classification_report(label_list, pred_list, digits = 4)
            # print("---------------分类报告\n", cp)
            return running_loss / len(iter), correct, total

