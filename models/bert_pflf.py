import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from models.bert_base import Bert_base

class Bert_pflf(Bert_base):
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

        batch_size, _ = nn_features.size()
        orthogonal_features = torch.zeros(batch_size, _).cuda()
        for i in range(batch_size):
            orthogonal_features[i] = self.get_orthogonal_vector(nn_features[i], ai_features[i])
        fusion_features = torch.cat((nn_features, orthogonal_features), dim=-1)
        fusion_features = self.dropout(fusion_features)
        logits = self.classifier(fusion_features)
        return logits, fusion_features

    @staticmethod
    def get_orthogonal_vector(main_vector, sub_vector):
        projection_vector = main_vector * torch.dot(main_vector, sub_vector) / torch.dot(main_vector, main_vector)
        orthogonal_vector = sub_vector - projection_vector
        return orthogonal_vector