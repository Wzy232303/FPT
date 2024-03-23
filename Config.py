import torch
import os
from torch import nn

class config(object):
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.slide_window = args.slide_window
        self.max_len = args.max_len
        self.seq_num = int(self.max_len / self.slide_window)
        self.learning_rate = args.lr
        self.loss_func = nn.CrossEntropyLoss()
        self.warm_up_ratio = args.warm_up_ratio
        self.seed = args.sample_seed
        self.num_labels = args.num_labels
        self.features_dim = args.features_dim
        self.prompt_len = args.prompt_len
        self.verb = args.verbalizer
        self.template_num = args.template_num
        if args.do_fewshot:
            self.k = args.k ## few-shot
            self.num_epochs = args.epochs
        else:
            self.num_epochs = int(args.epochs // 10)
            self.k = 'full'
        # self.device = torch.device('cuda:' + args.device if torch.cuda.is_available() else 'cpu')
        self.result_save_path = "-" + args.model_name + '_' + args.dataset_name + "_" + args.times + str(self.k) + ".txt"
        self.model_save_name = "-" + args.model_name + '_' + args.dataset_name + "_" + args.times + "_" + str(self.k) + "_" + str(args.sample_seed) + " " + str(args.train_seed)
        self.description = "batchsize: {}, lr:{}, k:{}, sample_seed:{}, train_seed:{}, random_feature:{}, prompt_len:{}, template_num: {}, sc:{}" \
                .format(self.batch_size, self.learning_rate, str(self.k),  args.sample_seed, args.train_seed, str(args.randomfeature), str(args.prompt_len), str(self.template_num), str(args.calibration))
