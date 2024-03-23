import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from Config import config

class Baserunner(object):
    def __init__(self,
                model,
                trainiter,
                deviter,
                testiter,
                args) -> None:
        self.model = model
        self.trainiter = trainiter
        self.deviter = deviter
        self.testiter = testiter
        self.config = config(args)
        self.min_loss = 10000
        self.randomfeature = args.randomfeature
        self.no_improve = 0

    def train(self):
        self.model.train()
        self.model.zero_grad()
        optimizer, scheduler = self.get_optimizer()
        for epoch in range(self.config.num_epochs):
            self.train_epoch(optimizer, scheduler, epoch)
            loss, _, _ = self.evaluate(self.deviter)
            if loss < self.min_loss:
                self.min_loss = loss
                torch.save(self.model.state_dict(), '/root/autodl-tmp/.autodl/model_save' + self.config.model_save_name + '.pth')
                print('save successfullly')
                self.no_improve = 0
            self.no_improve += 1
            if self.no_improve > 10:
                print("early stop")
                break
        self.test()

    def train_epoch(self, optimizer, scheduler, epoch, is_fewshot=True):
        self.model.train()
        with tqdm(total=len(self.trainiter), desc='Epoch {}'.format(epoch + 1), ncols=100) as pbar:
            for batch in self.trainiter:
                input_ids, token_type_ids, attention_mask, label = self.to_gpu(batch)
                # with autocast():
                input = input_ids, token_type_ids, attention_mask
                outputs = self.model(input)
                logits = outputs
                loss = self.config.loss_func(logits, label)
                pbar.update(1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                optimizer.step()
                scheduler.step()

    def test(self):
        self.model.load_state_dict(torch.load('/root/autodl-tmp/.autodl/model_save' + self.config.model_save_name + '.pth'))
        self.model.eval()
        test_loss, correct, total = self.evaluate(self.testiter)
        print('test loss: {:.6f} Accuracy on {} set: {:.3f}% [{}/{}]'.format(
            test_loss, self.config.dataset_name, 100 * correct / total, correct, total
        ))
        with open('result_final/' + self.config.result_save_path + 'calculation', 'a', encoding='utf-8') as f:
            f.write(self.config.description + ":")
            f.write(str(correct) + "\n") 
        f.close()

    def evaluate(self, iter):
        correct = 0
        total = 0
        with torch.no_grad():
            running_loss = 0
            for batch in iter:
                input_ids, token_type_ids, attention_mask, label = batch
                # with autocast():
                input = input_ids, token_type_ids, attention_mask
                logits = self.model(input)
                loss = self.config.loss_func(logits, label)
                pred = torch.max(logits, dim=1)[1]
                correct += (pred == label).sum().item()
                total += label.size(0)
                running_loss += loss.item()
            return running_loss / len(iter), correct, total

    def get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        num_training_steps = len(self.trainiter) * self.config.num_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(num_training_steps * self.config.warm_up_ratio), num_training_steps=num_training_steps
        )
        return optimizer, scheduler


    def write(self, time):
        if time == 0: 
            with open('result_final/' + self.config.result_save_path, 'a', encoding='utf-8') as f: # 先写入模型的参数细节
                f.write(self.config.description + "\n")
        elif time == 1: 
            with open('result_final/' + self.config.result_save_path + 'calculation', 'a', encoding='utf-8') as f: # 先写入模型的参数细节
                f.write(self.config.description + ":")
                f.write(str(self.max_correct) + "\n")   
        else:
            pass

