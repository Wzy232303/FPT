from data_utils.BaseDataprocessor import BaseDataProcessor
from data_utils.FeatureDataprocessor import FeatureDataProcessor
from data_utils.PromptDataprocessor import PromptDataProcessor
from data_utils.PromptFeatureDataprocessor import PromptFeatureDataprocessor
from models.bert_ptuning import Bert_ptuning
from models.bert_ftuning import Bert_ftuning
from models.bert_manual import Bert_manual
from models.bert_base import Bert_base
from models.bert_prompttuning import Bert_prompttuning
from models.bert_lf import Bert_lf
from models.bert_pflf import Bert_pflf
from models.bert_hybird_ftuning import Bert_hybird_ftuning
from models.bert_hybird_prompttuning import Bert_hybird_prompttuning
from runer.RArunner import RArunner
from verbalizers.softverb import SoftVerbalizer
from verbalizers.protoverb import ProtoClVerbalizer
from arguments import get_args_parser
import torch
import os
import numpy as np

def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


def run(args):
    same_seeds(args.train_seed)
    if args.model_name == 'bert_lf':
        dataprocessor = FeatureDataProcessor(args)
        verbalizer = SoftVerbalizer(args)
        model = Bert_lf(args=args, verbalizer=verbalizer)
    elif args.model_name == 'bert_pflf':
        dataprocessor = FeatureDataProcessor(args)
        verbalizer = SoftVerbalizer(args)
        model = Bert_pflf(args=args, verbalizer=verbalizer)
    elif args.model_name == 'bert_base':
        dataprocessor = FeatureDataProcessor(args)
        verbalizer = SoftVerbalizer(args)
        model = Bert_base(args=args, verbalizer=verbalizer)
    elif args.model_name == 'bert_ftuning':
        dataprocessor = FeatureDataProcessor(args)
        if args.verbalizer == 'proto' or args.verbalizer == 'proto-base':
            verbalizer = ProtoClVerbalizer(args)
        elif args.verbalizer == 'soft':
            verbalizer = SoftVerbalizer(args)
        model = Bert_ftuning(args=args, verbalizer=verbalizer)
    elif args.model_name == 'bert_hybird_ftuning':
        dataprocessor = PromptFeatureDataprocessor(args)
        if args.verbalizer == 'proto' or args.verbalizer == 'proto-base':
            verbalizer = ProtoClVerbalizer(args)
        elif args.verbalizer == 'soft':
            verbalizer = SoftVerbalizer(args)
        model = Bert_hybird_ftuning(args=args, verbalizer=verbalizer)
    elif args.model_name == 'bert_manual':
        dataprocessor = PromptDataProcessor(args)
        if args.verbalizer == 'proto' or args.verbalizer == 'proto-base':
            verbalizer = ProtoClVerbalizer(args)
        elif args.verbalizer == 'soft':
            verbalizer = SoftVerbalizer(args)
        model = Bert_manual(args=args, verbalizer=verbalizer)
    elif args.model_name == 'bert_ptuning':
        dataprocessor = PromptDataProcessor(args)
        if args.verbalizer == 'proto' or args.verbalizer == 'proto-base':
            verbalizer = ProtoClVerbalizer(args)
        elif args.verbalizer == 'soft':
            verbalizer = SoftVerbalizer(args)
        model = Bert_ptuning(args=args, verbalizer=verbalizer)
    elif args.model_name == 'bert_prompttuning':
        dataprocessor = BaseDataProcessor(args)
        if args.verbalizer == 'proto' or args.verbalizer == 'proto-base':
            verbalizer = ProtoClVerbalizer(args)
        elif args.verbalizer == 'soft':
            verbalizer = SoftVerbalizer(args)
        model = Bert_prompttuning(args=args, verbalizer=verbalizer)
    elif args.model_name == 'bert_hybird_prompttuning':
        dataprocessor = PromptDataProcessor(args)
        if args.verbalizer == 'proto' or args.verbalizer == 'proto-base':
            verbalizer = ProtoClVerbalizer(args)
        elif args.verbalizer == 'soft':
            verbalizer = SoftVerbalizer(args)
        
        model = Bert_hybird_prompttuning(args=args, verbalizer=verbalizer)
    trainiter, deviter, testiter = dataprocessor.get_iter()
    model = model.cuda()
    model=torch.nn.DataParallel(model)
    runner = RArunner(model, trainiter, deviter, testiter, args)
    runner.train()

if __name__ == '__main__':
    args = get_args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids.replace('_', ',')
    run(args)