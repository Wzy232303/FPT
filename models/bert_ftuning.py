import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from Config import config
from models.bert_base import Bert_base
import numpy as np

class Bert_ftuning(Bert_base):
    def __init__(self, args, verbalizer=None) -> None:
        super().__init__(args, verbalizer)
        self.feature_dim = sum(self.config.features_dim)
        self.orthogonalize = args.orthogonalize
        self.prefix_transes = nn.ModuleList([nn.Sequential(
                                nn.Linear(self.feature_dim, self.modelconfig.hidden_size),
                                nn.Tanh(),
                                nn.Linear(self.modelconfig.hidden_size, self.modelconfig.hidden_size)
                            ) for _ in range(self.config.prompt_len)])


    def get_feature_prompt(self, c, w, s , p):
        prefix = torch.cat((c, w, s, p), dim=-1) # batch_size * feature_dim
        # prefix = self.trans(prefix)
        prefixs = torch.cat([l(prefix).unsqueeze(1) for l in self.prefix_transes], dim=1)
        if self.orthogonalize:
            prefixs = self.orthogonalize_vectors(prefixs)
        return prefixs## bs * prmoptlen * hidden size
    
    def orthogonalize_vectors(self, tensor):
        batch_size, num_vectors, vector_dim = tensor.shape
        # 将每个样本的10个向量拆分出来
        vectors = torch.unbind(tensor, dim=0)
        # print(vectors[0].shape)
        orthogonal_vectors = []
        for vector in vectors:
            Q, R = torch.qr(vector.t())
            orthogonal_vector = Q.t()[:num_vectors]
            orthogonal_vectors.append(orthogonal_vector)

        orthogonal_vectors = torch.stack(orthogonal_vectors, dim=0)
        return orthogonal_vectors
    


    def forward(self, inputs, *args):
        # input_ids = [batch_size, seq_len]
        input_ids, token_type_ids, attention_mask = inputs
        c, w, s, p = args
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
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.model.pooler.dense(first_token_tensor)
        pooled_output = self.model.pooler.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, pooled_output, prompts
    

