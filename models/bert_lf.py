import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from models.bert_base import Bert_base

class Bert_lf(Bert_base):
    def __init__(self, args, verbalizer=None) -> None:
        super().__init__(args, verbalizer)
        self.trans_ai = nn.Linear(sum(self.config.features_dim), 384)
        self.trans_nn = nn.Linear(self.modelconfig.hidden_size, 384)

    def forward(self, input, *args):
        # input_ids = [batch_size, seq_num, seq_len]
        input_ids, token_type_ids, attention_mask = input
        batch_size, seq_len = input_ids.size()
        # mask_token_index = mask_token_index.view(batch_size * seq_num, seq_len)
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        c, w, s, p = args
        ai_features = F.tanh(self.trans_ai(torch.cat((c, w, s, p), dim=-1)))
        nn_features = F.tanh(self.trans_nn(pooled_output))
        fusion_features = torch.cat((ai_features, nn_features), dim=-1)
        fusion_features = self.dropout(fusion_features)
        logits = self.classifier(fusion_features)
        return logits, fusion_features