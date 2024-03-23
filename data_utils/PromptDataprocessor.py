from torch.utils.data import DataLoader, TensorDataset
from data_utils.BaseDataprocessor import BaseDataProcessor
from data_utils.templates.Template import templates
import random
import ast
import os
import torch
from Config import config

class PromptDataProcessor(BaseDataProcessor):
    def __init__(self, args) -> None:
        self.templates = templates(args)
        BaseDataProcessor.__init__(self, args)
        

    def get_inputs(self, type)->tuple:
        """
        return tensor tuple
        """
        path = os.path.join(self.data_path, type + '_dict.txt')
        labels, dataset = [], []
        with open(path, 'r', encoding='utf-8') as f_dataset:
            for data in f_dataset:
                text, label = self.process_dataline(data)
                labels.append(label)
                dataset.append(text)
        if self.few_shot and type != 'test':
            data_zip = list(zip(dataset, labels))
            dataset, labels = self.get_few_shot(datazip=data_zip)
        if self.is_seg:
            dataset = [i for item in dataset for i in item]
        # print(dataset[0][:20])
        inputs = self.tokenizer(
            list(dataset),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.config.slide_window + 2,
            is_split_into_words= False,
            return_tensors = 'pt')
        mask_token_index = inputs.input_ids == self.tokenizer.mask_token_id ## [datanum*seq_num, slide_window + 2]
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
        return input_ids, token_type_ids, attention_mask, labels, mask_token_index

    def process_dataline(self, data: str)->tuple:
        """
        base: return text + label
        seg: return seg_text[list] + label
        """
        data_dict = ast.literal_eval(data)
        if 'prompt' not in self.config.dataset_name:
            label = int(data_dict['label'])
        else:
            label = float(data_dict['label'])
        text = data_dict['text']
        if not self.zh:
            text = self.convert_entext_to_words(text)
            if not self.is_seg:
                text = text[0: self.config.slide_window - self.templates.template_len]
        if self.is_seg:
            cut_text = self.cut_text(text)
            return cut_text, label
        text = self.templates.wrap_text(text)
        return text, label 

    def cut_text(self, text: str or list)->list:
        """
        do_seg
        """
        cut_text = []
        for i in range(0, len(text), self.config.slide_window - self.templates.template_len):
            cut_text.append(text[i: i + self.config.slide_window - self.templates.template_len])
        cut_text = cut_text[0: self.config.seq_num]
        for i in range(0, self.config.seq_num - len(cut_text)):
            if self.zh:
                cut_text.append("")
            else:
                cut_text.append([])
        cut_text = self.templates.wrap_text(cut_text)
        return cut_text
