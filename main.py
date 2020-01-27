import argparse
from model_handler import ModelHandler
from utils.download import download_model

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', type = str, default = '/home/aniket/coqa/data/coqa.train.json', help = 'training dataset file')
parser.add_argument('--devset', type = str, default = '/home/aniket/coqa/data/coqa.dev.json', help = 'development dataset file')
parser.add_argument('--model_name', type = str, default = 'BERT', help = '[BERT|RoBERTa|DistilBERT|SpanBERT]')
parser.add_argument('--model_path', type = str, default = None, help = 'path to pretrained model')
parser.add_argument('--pretrained_dir', type = str, default = 'model')
parser.add_argument('--save_state_dir', type = str, default = None)

parser.add_argument('--cuda', type = str2bool, default = True, help = 'use gpu or not')
parser.add_argument('--debug', type = str2bool, default = True)

parser.add_argument('--n_history', type = int, default = 2, help = 'number of previous question to use as previous context')
parser.add_argument('--batch-size', type = int, default = 2)
parser.add_argument('--shuffle', type = str2bool, default = True)
parser.add_argument('--max_epochs', type = int, default = 20)
parser.add_argument('--lr', type = float, default = 2e-4)
parser.add_argument('--grad_clip', type = float, default = 1.0)
parser.add_argument('--verbose', type = int, default = 200, help = "print after verbose epochs")
parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")



args = vars(parser.parse_args())

if args['model_name'] == 'SpanBERT':
    download_model()
    args['model_path'] = 'tmp_' 


# TODO: cuda check

handler = ModelHandler(args)
handler.train()