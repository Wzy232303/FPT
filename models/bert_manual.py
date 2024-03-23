import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from Config import config
from models.bert_base import Bert_base

class Bert_manual(Bert_base):
    def __init__(self, args, verbalizer=None) -> None:
        super().__init__(args, verbalizer)

    
    def forward(self, inputs, *args):
        # input_ids = [batch_size, seq_len]
        input_ids, token_type_ids, attention_mask = inputs
        mask_token_index = args[-1]
        batch_size, seq_len = input_ids.size()
        hidden_states = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1] # [batch_size, seq_len, 768]
        mask_token_index = mask_token_index.unsqueeze(2).expand(batch_size, seq_len, self.modelconfig.hidden_size)
        hidden_states = torch.masked_select(hidden_states, mask_token_index).view(-1, self.modelconfig.hidden_size) #[batch_size,  768]
        logits = self.verbalizer.process_outputs(hidden_states)
        return logits, hidden_states

    

