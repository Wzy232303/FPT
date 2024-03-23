import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from Config import config
from models.bert_base import Bert_base
import numpy as np

class Bert_prompttuning(Bert_base):
    def __init__(self, args, verbalizer=None) -> None:
        super().__init__(args, verbalizer)
        self.prefix_tokens = torch.arange(self.config.prompt_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.config.prompt_len, self.modelconfig.hidden_size)
        # indices = np.random.permutation(range(5000))[:self.config.prompt_len]
        # init_weight = self.model.embeddings.word_embeddings.state_dict()["weight"][indices]
        # self.prefix_encoder._load_from_state_dict({"weight": init_weight},
        #                                      "", None, True, [], [], "")

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).cuda()
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts ## bs * prmoptlen * hidden size
    
    def forward(self, inputs, *args):
        # input_ids = [batch_size, seq_len]
        input_ids, token_type_ids, attention_mask = inputs
        # c, w, s, p, mask_token_index = args
        batch_size, seq_len = input_ids.size()
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1) # batch_size * seq_num, seq_len + 5 768
        prefix_attention_mask = torch.ones(batch_size, self.config.prompt_len).cuda()
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        hidden_states = self.model(inputs_embeds=inputs_embeds,attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1] # [batch_size, seq_len, 768]
        hidden_states = hidden_states[:, self.config.prompt_len:, :].contiguous() ## 去掉prefix
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.model.pooler.dense(first_token_tensor)
        pooled_output = self.model.pooler.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, pooled_output, prompts

    

