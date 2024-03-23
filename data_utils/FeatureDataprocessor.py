from data_utils.BaseDataprocessor import BaseDataProcessor
from torch.utils.data import DataLoader, TensorDataset
import random
import ast
import os
import torch
from Config import config

class FeatureDataProcessor(BaseDataProcessor):
    def __init__(self, args) -> None:
        self.is_randomfeature = args.randomfeature
        super().__init__(args)


    def get_inputs(self, type)->tuple:
        path = os.path.join(self.data_path, type + '_dict.txt')
        labels, dataset, features = [], [], []
        with open(path, 'r', encoding='utf-8') as f_dataset:
            for data in f_dataset:
                text, label, feature = self.process_dataline(data)
                labels.append(label)
                dataset.append(text)
                features.append(feature)
        # print(isinstance(dataset, list))
        if self.few_shot and type != 'test':
            data_zip = list(zip(dataset, labels, features))
            dataset, labels, features = self.get_few_shot(datazip=data_zip)
        # print(isinstance(dataset, list))
        if self.is_seg:
            dataset = [i for item in dataset for i in item]
        inputs = self.tokenizer(
            list(dataset),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.config.slide_window + 2,
            is_split_into_words = False,
            return_tensors = 'pt')
        input_ids = inputs['input_ids']
        if not self.BPE:
            token_type_ids = inputs['token_type_ids']
        else:
            token_type_ids = torch.zeros_like(input_ids)
        attention_mask = inputs['attention_mask']
        if self.is_seg:
            input_ids = input_ids.view(-1, self.config.seq_num, self.config.slide_window + 2)
            token_type_ids = token_type_ids.view(-1, self.config.seq_num, self.config.slide_window + 2)
            attention_mask = attention_mask.view(-1, self.config.seq_num, self.config.slide_window + 2)
        labels = torch.tensor(labels)
        if self.is_randomfeature:
            c_features = torch.rand(len(input_ids), self.config.features_dim[0])
            w_features = torch.rand(len(input_ids), self.config.features_dim[1])
            s_features = torch.rand(len(input_ids), self.config.features_dim[2])
            p_features = torch.rand(len(input_ids), self.config.features_dim[3])
        else:
            c_features = torch.tensor([feature[0] for feature in features])
            w_features = torch.tensor([feature[1] for feature in features])
            s_features = torch.tensor([feature[2] for feature in features])
            p_features = torch.tensor([feature[3] for feature in features])
        return input_ids, token_type_ids, attention_mask, labels, c_features, w_features, s_features, p_features                  

    def process_dataline(self, data: str)->tuple:
        """
        feature: return seg_text[list] + label + features[tuple]
        """
        data_dict = ast.literal_eval(data)
        if 'prompt' not in self.config.dataset_name:
            label = int(data_dict['label'])
        else:
            label = float(data_dict['label'])
        text = data_dict['text']
        feature = self.process_feature(data_dict)
        if not self.zh:
            text = self.convert_entext_to_words(text)
        if self.is_seg:
            cut_text = self.cut_text(text)
            return cut_text, label, feature
        text = self.tokenizer.convert_tokens_to_string(text)
        # print(text)
        return text, label, feature

    def process_feature(self, data_dict: dict)->tuple:
        """
        process features: 4 c w s p
        """
        c_feature = list(data_dict['char_features'])
        w_feature = list(data_dict['word_features'])
        s_feature = list(data_dict['sentence_features'])
        p_feature = list(data_dict['paragraph_features'])
        feature = (c_feature, w_feature, s_feature, p_feature)
        return feature