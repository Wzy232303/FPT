from torch.utils.data import DataLoader, TensorDataset
import random
import ast
import os
import torch
from Config import config

class BaseDataProcessor(object):
    def __init__(self, args) -> None:
        self.config = config(args)
        self.tokenizer = args.tokenizer
        self.BPE = (args.model_name.split('_')[0] == 'roberta')
        # print(self.BPE)
        self.data_path = os.path.join('datasets', self.config.dataset_name)
        self.few_shot = args.do_fewshot ## whether do few-shot
        self.is_seg = args.do_seg ## whether seg the text into several short texts.
        self.zh = args.zh ## english dataset or chinese
        self.train_data = self.to_cuda(self.get_inputs('train'))
        self.dev_data = self.to_cuda(self.get_inputs('dev'))
        self.test_data = self.to_cuda(self.get_inputs('test'))

        

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
        inputs = self.tokenizer(
            list(dataset),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.config.slide_window + 2,
            is_split_into_words= False,
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
        return (input_ids, token_type_ids, attention_mask, labels)
        

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
        if (not self.zh):
            text = self.convert_entext_to_words(text)
        if self.is_seg:
            cut_text = self.cut_text(text)
            return cut_text, label
        text = self.tokenizer.convert_tokens_to_string(text)
        return text, label 

    def convert_entext_to_ids(self, text: str)->tuple:
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def convert_entext_to_words(self, text: str)->tuple:
        return self.tokenizer.tokenize(text)

    def cut_text(self, text: str or list)->list:
        """
        do_seg
        """
        cut_text = []
        for i in range(0, len(text), self.config.slide_window):
            cut_text.append(text[i: i + self.config.slide_window])
        cut_text = cut_text[0: self.config.seq_num]
        for i in range(0, self.config.seq_num - len(cut_text)):
            if self.zh:
                cut_text.append("")
            else:
                cut_text.append([])
        if isinstance(text, list):
            cut_text = [self.tokenizer.convert_tokens_to_string(t) for t in cut_text]
        return cut_text

    def get_iter(self, ):
        # assert all(self.train_data[0].size(0) == tensor.size(0) for tensor in self.train_data)
        train_iter = DataLoader(TensorDataset(*self.train_data), shuffle=True, batch_size=self.config.batch_size)
        dev_iter = DataLoader(TensorDataset(*self.dev_data), shuffle=False, batch_size=self.config.batch_size)
        test_iter = DataLoader(TensorDataset(*self.test_data), shuffle=False, batch_size=self.config.batch_size * 8)
        # print(len(train_iter), len(test_iter))
        return train_iter, dev_iter, test_iter

    def get_few_shot(self, datazip, balance = True):
        """
        if balance == True:
            label is balanced
        else
            label is random
        """
        random.seed(self.config.seed)
        if self.config.k > len(datazip):
            assert len(datazip) == 30
            return zip(*datazip)
        if balance:
            datazip0 = [item for item in datazip if item[1] == 0]
            datazip1 = [item for item in datazip if item[1] == 1]
            datazip2 = [item for item in datazip if item[1] == 2]
            datazip3 = [item for item in datazip if item[1] == 3]
            datazip4 = [item for item in datazip if item[1] == 4]
            # print(datazip[0][1])
            return zip(*(random.sample(datazip0, int(self.config.k/5)) + random.sample(datazip1, int(self.config.k/5)) + random.sample(datazip2, int(self.config.k/5)) + random.sample(datazip3, int(self.config.k/5)) + random.sample(datazip4, int(self.config.k/5))))
        else:
            return zip(*random.sample(datazip, self.config.k))

    def to_cuda(self, data):
        data = list(data)
        for index, _ in enumerate(list(data)):
            data[index] = data[index].cuda()
        return tuple(data)