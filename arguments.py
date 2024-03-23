import argparse
import torch
import transformers
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,\
                         AlbertTokenizer, AlbertConfig, AlbertModel, \
                         BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.albert.modeling_albert import AlbertMLMHead


def get_args_parser():
    parser = argparse.ArgumentParser(description="Command line interface for Prompt-based RA.")
    parser.add_argument('--dataset_name', type=str, default='ChineseRA')
    parser.add_argument('--model_name', type=str, default='bert_base')
    parser.add_argument('--device_ids', type=str, default='0_1')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--loss', type=str, default="CrossEntropy")
    parser.add_argument('--verbalizer', type=str, default="mannul")

    parser.add_argument('--sample_seed', type=int, default=1)
    parser.add_argument('--train_seed', type=int, default=1)
    parser.add_argument('--k', type=int, default=5, help='num of sample examples each class, if do_fewshot')
    parser.add_argument('--prompt_len', type=int, default=0, help='length of soft prompt')
    # parser.add_argument('--slide_window', type=int, default=510, help='length of seg text, if do_seg')
    parser.add_argument('--max_len', type=int, default=4048)
    parser.add_argument('--warm_up_ratio', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=30) ## train epoch if do not few shot, epoch / 10
    parser.add_argument('--template_num', type=str, default='0') ## index of template
    parser.add_argument('--times', type=str, default='1')

    parser.add_argument('--zh', action = 'store_true', default=False)
    parser.add_argument('--do_seg', action = 'store_true', default=False)
    parser.add_argument('--do_frozen', action = 'store_true', default=False)
    parser.add_argument('--randomfeature', action = 'store_true', default=False)
    parser.add_argument('--do_fewshot', action = 'store_true', default=False)
    parser.add_argument('--calibration', action = 'store_true', default=False)
    parser.add_argument('--orthogonalize', action = 'store_true', default=False)
    args = parser.parse_args()
    args.tokenizer = BertTokenizer
    args.num_labels = 5
    if args.dataset_name == "Cambridge":
        args.prompt_len += 20
    args.slide_window = 510 - args.prompt_len
    if args.model_name.split('_')[0] == 'bert':
        if args.zh:
            args.features_dim = [72, 145, 60, 5]
            args.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        else:
            args.features_dim = [28, 109, 56, 14]
            args.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.model_name.split('_')[0] == 'roberta':
        if args.zh:
            args.features_dim = [72, 145, 60, 5] ## 6 * 47 282
            args.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', )
        else:
            args.features_dim = [28, 109, 56, 14] ## 9 * 23 207
            args.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    else:
        pass
    if args.zh:
        if args.template_num == '0':
            args.replace_index = [1,2,3,7,8,9]
        elif args.template_num == '1':
            args.replace_index = [1,2,3,7,8,9]
        elif args.template_num == '2':
            args.replace_index = [1,2,3,4,5,9,10,11]
        elif args.template_num == '3':
            args.replace_index = [1,2,7,9,10,11]
        else:
            pass
    else:
        if args.template_num == '0':
            args.replace_index = [1,3,4]
        elif args.template_num == '1':
            args.replace_index = [1,3,4]
        elif args.template_num == '2':
            args.replace_index = [1,2,3,5,6]
        elif args.template_num == '3':
            args.replace_index = [1,3,4]
        else:
            pass
    return args