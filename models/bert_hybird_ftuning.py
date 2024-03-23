"""
soft + anchor words prompt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from Config import config
from models.bert_base import Bert_base
import numpy as np

class Bert_hybird_ftuning(Bert_base):
    def __init__(self, args, verbalizer=None) -> None:
        super().__init__(args, verbalizer)
        # self.avg_pooling = nn.AvgPool1d(kernel_size=(self.config.seq_num, 1), stride=1)
        self.orthogonalize = args.orthogonalize
        self.feature_dim = sum(self.config.features_dim)
        self.prefix_transes = nn.ModuleList([nn.Sequential(
                                nn.Linear(self.feature_dim, self.modelconfig.hidden_size),
                                nn.Tanh(),
                                nn.Linear(self.modelconfig.hidden_size, self.modelconfig.hidden_size)
                            ) for _ in range(self.config.prompt_len)])


    def get_feature_prompt(self, c, w, s, p):
        prefix = torch.cat((c, w, s, p), dim=-1) # batch_size * feature_dim
        prefixs = torch.cat([l(prefix).unsqueeze(1) for l in self.prefix_transes], dim=1)
        return  prefixs## bs * prmoptlen * hidden size
    
    
    def forward(self, inputs, *args):
        # input_ids = [batch_size, seq_len]
        input_ids, token_type_ids, attention_mask = inputs
        c, w, s, p, mask_token_index = args
        batch_size, seq_len = input_ids.size()
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_feature_prompt(c, w, s, p)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1) # batch_size * seq_num, seq_len + 5 768
        prefix_attention_mask = torch.ones(batch_size, self.config.prompt_len).cuda()
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        hidden_states = self.model(inputs_embeds=inputs_embeds,attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1] # [batch_size, seq_len, 768]
        hidden_states = hidden_states[:, self.config.prompt_len:, :].contiguous() ## 去掉prefix
        mask_token_index = mask_token_index.unsqueeze(2).expand(batch_size, seq_len, self.modelconfig.hidden_size)
        mask_hidden_states = torch.masked_select(hidden_states, mask_token_index).view(-1, self.modelconfig.hidden_size) #[batch_size,  768]
        # features_states = nn.functional.avg_pool2d(prompts, kernel_size=(self.config.prompt_len, 1), stride=1).squeeze(1)
        logits = self.verbalizer.process_outputs(mask_hidden_states)
        return logits, mask_hidden_states, prompts
    

