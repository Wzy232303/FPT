from torch.utils.data import DataLoader, TensorDataset
from data_utils.PromptDataprocessor import PromptDataProcessor
from data_utils.FeatureDataprocessor import FeatureDataProcessor
from data_utils.templates.Template import templates
import random
import ast
import os
import torch
from Config import config

class PromptFeatureDataprocessor(PromptDataProcessor, FeatureDataProcessor):
    def __init__(self, args) -> None:
        self.is_randomfeature = args.randomfeature
        PromptDataProcessor.__init__(self, args)
        FeatureDataProcessor.__init__(self, args)
        # print(self.templates.template_len)
        
        
    
    def process_dataline(self, data: str) -> tuple:
        """
        return seg_text[list] + label + features[tuple]
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
            if not self.is_seg:
                text = text[0: self.config.slide_window - self.templates.template_len]
        if self.is_seg:
            cut_text = self.cut_text(text)
            return cut_text, label, feature
        text = self.templates.wrap_text(text)
        # print(text[-10: ])
        return text, label, feature

    def get_inputs(self, type) -> tuple:
        path = os.path.join(self.data_path, type + '_dict.txt')
        labels, dataset, features = [], [], []
        with open(path, 'r', encoding='utf-8') as f_dataset:
            for data in f_dataset:
                text, label, feature = self.process_dataline(data)
                labels.append(label)
                dataset.append(text)
                features.append(feature)
        if self.few_shot and type != 'test':
            data_zip = list(zip(dataset, labels, features))
            # print(len(data_zip))
            dataset, labels, features = self.get_few_shot(datazip=data_zip)
        if self.is_seg:
            dataset = [i for item in dataset for i in item]
        # print(dataset[0])
        # print(dataset[1])
        inputs = self.tokenizer.batch_encode_plus(
            list(dataset),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.config.slide_window + 2,
            is_split_into_words= False,
            return_tensors = 'pt')
        # for input_id in inputs.input_ids:
        #     with open('test.txt', 'a', encoding='utf-8') as f:
        #         f.write(self.tokenizer.decode(input_id))
        mask_token_index = inputs.input_ids == self.tokenizer.mask_token_id ## [datanum*seq_num, slide_window + 2]
        # print(mask_token_index)
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
            mask_token_index = mask_token_index.view(-1, self.config.seq_num, self.config.slide_window + 2)
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
        return input_ids, token_type_ids, attention_mask, labels, c_features, w_features, s_features, p_features, mask_token_index
