import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from Config import config

class Bert_base(nn.Module):
    """
    bert base
    """
    def __init__(self, 
                args,
                verbalizer=None) -> None:
        super(Bert_base, self ).__init__() 
        if args.zh:   
            self.model = BertModel.from_pretrained('bert-base-chinese')
            self.modelconfig = BertConfig.from_pretrained('bert-base-chinese')
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.modelconfig = BertConfig.from_pretrained('bert-base-uncased')
        self.embeddings = self.model.embeddings
        self.config = config(args)
        if args.do_frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        self.num_labels = args.num_labels
        self.classifier = nn.Linear(self.modelconfig.hidden_size, self.num_labels)
        self.verbalizer = verbalizer
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input, *args):
        input_ids, token_type_ids, attention_mask = input
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, torch.tensor(0)
        