import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from Config import config
from models.bert_base import Bert_base
import numpy as np

class Bert_ptuning(Bert_base):
    def __init__(self, args, verbalizer=None) -> None:
        super().__init__(args, verbalizer)
        self.replace_indices = args.replace_index
        self.prefix_tokens = torch.arange(self.config.prompt_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.config.prompt_len, self.modelconfig.hidden_size)
        self.lstm_head = torch.nn.LSTM(input_size=self.modelconfig.hidden_size,
                                           hidden_size=self.modelconfig.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(2 * self.modelconfig.hidden_size, self.modelconfig.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.modelconfig.hidden_size, self.modelconfig.hidden_size)
                                    )
       

    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).cuda()
        prompts = self.prefix_encoder(prefix_tokens) # [batch_size, seq_len, 2 * hidden_dim]
        # print(prompts.size())
        prompts = self.lstm_head(prompts)[0]
        if self.config.prompt_len == 1:
                prompts = self.mlp_head(prompts)
        else:
                prompts = self.mlp_head(prompts).squeeze()
        return prompts ## prmoptlen * hidden size
    

    
    def forward(self, inputs, *args):
        # input_ids = [batch_size, seq_len]
        input_ids, token_type_ids, attention_mask = inputs
        mask_token_index = args[-1]
        
        batch_size, seq_len = input_ids.size() 
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        ) # bs， seq_len, hidden, dim
        prompts = self.get_prompt(batch_size)
        # print(prompts.size())
        mask = torch.ones(seq_len, dtype=torch.bool).cuda()
        seq_len = seq_len - len(self.replace_indices) + self.config.prompt_len
        mask[self.replace_indices] = False
        raw_embedding = raw_embedding[:, mask, :]
        attention_mask = attention_mask[:, mask]
        mask_token_index = mask_token_index[:, mask]
        raw_embedding = torch.cat((raw_embedding[:, :1, :], prompts, raw_embedding[:, 1:, :]), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.config.prompt_len).cuda()
        prefix_mask_token_index = torch.zeros((batch_size, self.config.prompt_len), dtype=torch.bool).cuda()
        # print(attention_mask.size())
        # print(attention_mask[:,:1].size())
        attention_mask = torch.cat((attention_mask[:,:1],prefix_attention_mask, attention_mask[:,1:]), dim=1)
        mask_token_index = torch.cat((mask_token_index[:,:1],prefix_mask_token_index , mask_token_index[:,1:]), dim=1)
        hidden_states = self.model(inputs_embeds=raw_embedding,attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1] # [batch_size, seq_len, 768]
        mask_token_index = mask_token_index.unsqueeze(2).expand(batch_size, seq_len, self.modelconfig.hidden_size)
        hidden_states = torch.masked_select(hidden_states, mask_token_index).view(-1, self.modelconfig.hidden_size) #[batch_size,  768]
        logits = self.verbalizer.process_outputs(hidden_states)
        return logits, hidden_states

    

