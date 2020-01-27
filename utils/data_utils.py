import json
import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter, defaultdict
import numpy as np
from random import shuffle
import math
import textacy.preprocessing.replace as rep
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_sm')

def prepare_datasets(config, tokenizer_model):
    tokenizer = tokenizer_model[1].from_pretrained(tokenizer_model[2])
    trainset = CoQADataset(config['trainset'])
    trainset.chunk_paragraphs(tokenizer, config['model_name'])
    trainloader = CustomDataLoader(trainset, config['batch_size'])
    devset = CoQADataset(config['devset'])
    devset.chunk_paragraphs(tokenizer, config['model_name'])
    devloader = CustomDataLoader(devset, config['batch_size'])
    return trainloader, devloader, tokenizer
def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)

def preprocess(text):
    text = ' '.join(text)
    temp_text = rep.replace_currency_symbols(text, replace_with = '_CUR_')
    temp_text = rep.replace_emails(temp_text, replace_with = '_EMAIL_')
    temp_text = rep.replace_emojis(temp_text, replace_with='_EMOJI_')
    temp_text = rep.replace_hashtags(temp_text, replace_with='_TAG_')
    temp_text = rep.replace_numbers(temp_text, replace_with='_NUMBER_')
    temp_text = rep.replace_phone_numbers(temp_text, replace_with = '_PHONE_')
    temp_text = rep.replace_urls(temp_text, replace_with = '_URL_')
    temp_text = rep.replace_user_handles(temp_text, replace_with = '_USER_')

    doc = nlp(temp_text)
    tokens = []
    for t in doc:
        tokens.append(t.text)
    return tokens

class CoQADataset(Dataset):
    """CoQA dataset."""

    def __init__(self, filename):
        #timer = Timer('Load %s' % filename)
        self.filename = filename
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
        dataset = read_json(filename)
        for paragraph in tqdm(dataset['data']):
            #print(paragraph)
            history = []
            for qas in paragraph['qas']:
                qas['paragraph_id'] = len(self.paragraphs)
                temp = []
                n_history = len(history) #if config['n_history'] < 0 else min(config['n_history'], len(history))
                if n_history > 0:
                    for i, (q, a) in enumerate(history[-n_history:]):
                        q1 = preprocess(q)
                        a1 = preprocess(a)
                        d = n_history - i
                        temp.append('<Q{}>'.format(d))
                        temp.extend(q1)
                        temp.append('<A{}>'.format(d))
                        temp.extend(a1)
                temp.append('<Q>')
                temp.extend(qas['annotated_question']['word'])
                history.append((qas['annotated_question']['word'], qas['annotated_answer']['word']))
                qas['annotated_question']['word'] = temp
                self.examples.append(qas)
                question_lens.append(len(qas['annotated_question']['word']))
                paragraph_lens.append(len(paragraph['annotated_context']['word']))
                for w in qas['annotated_question']['word']:
                    self.vocab[w] += 1
                for w in paragraph['annotated_context']['word']:
                    self.vocab[w] += 1
                for w in qas['annotated_answer']['word']:
                    self.vocab[w] += 1
            self.paragraphs.append(paragraph)
        print('Load {} paragraphs, {} examples.'.format(len(self.paragraphs), len(self.examples)))
        print('Paragraph length: avg = %.1f, max = %d' % (np.average(paragraph_lens), np.max(paragraph_lens)))
        print('Question length: avg = %.1f, max = %d' % (np.average(question_lens), np.max(question_lens)))
        #timer.finish()
        self.chunked_examples = []
        

    def chunk_paragraphs(self, tokenizer, model_name):
        c_unknown = 0
        c_known = 0
        dis = 0
        for i, ex in tqdm(enumerate(self.examples)):
            question_length = len(ex['annotated_question']['word'])
            if question_length > 350: # TODO provide from config
                continue
            doc_length_available = 512 - question_length - 3
            if model_name == 'RoBERTa':
                doc_length_available = doc_length_available - 3
            
            paragraph = self.paragraphs[ex['paragraph_id']]['annotated_context']['word']
            paragraph = preprocess(paragraph)
            if model_name != 'RoBERTa' and model_name != 'SpanBERT':
                paragraph = [p.lower() for p in paragraph]
            paragraph_length = len(paragraph)
            start_offset = 0
            doc_spans = []
            while start_offset < paragraph_length:
                length = paragraph_length - start_offset
                if length > doc_length_available:
                    length = doc_length_available - 1
                    doc_spans.append([start_offset, length, 1])
                else:
                    doc_spans.append([start_offset, length, 0])
                if start_offset + length == paragraph_length:
                    break
                start_offset += length
            for spans in doc_spans:
                segment_ids = []
                tokens = []
                if model_name == 'RoBERTa':
                    tokens.append('<s>')
                for q in ex['annotated_question']['word']:
                    segment_ids.append(0)
                    if model_name == 'RoBERTa' or model_name == 'SpanBERT':
                        tokens.append(q)
                        tokenizer.add_tokens([q])
                    else:
                        tokens.append(q.lower())
                        tokenizer.add_tokens([q.lower()])

                if model_name == 'RoBERTa':
                    tokens.extend(['</s>', '</s>'])
                else:    
                    tokens.append('[SEP]')
                    segment_ids.append(0)
                
                tokenizer.add_tokens(paragraph[spans[0]:spans[0] + spans[1]])
                tokens.extend(paragraph[spans[0]:spans[0] + spans[1]])
                segment_ids.extend([1] * spans[1])
                yes_index = len(tokens)
                tokens.append('yes')
                segment_ids.append(1)
                no_index = len(tokens)
                tokens.append('no')
                segment_ids.append(1)

                if spans[2] == 1:
                    tokens.append('<unknown>')
                    tokenizer.add_tokens(['<unknown>'])
                    segment_ids.append(1)
                if model_name == 'RoBERTa':
                    tokens.append('</s>')
                input_mask = [1] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                converted_to_string = tokenizer.convert_ids_to_tokens(input_ids)
                input_ids.extend([0]*(512 - len(tokens)))
                input_mask.extend([0] * (512 - len(tokens)))
                segment_ids.extend([0] * (512 - len(tokens)))

                start = ex['answer_span'][0]
                end = ex['answer_span'][1]

                if start >= spans[0] and end <= spans[1]:
                    c_known+=1
                    start = question_length + 1 + start
                    end = question_length + 1 + end
                    
                else:
                    c_unknown+=1
                    start = len(tokens) - 1
                    end = len(tokens) - 1
                if ex['answer'] == 'yes' and tokens[start]!='yes':
                    start = yes_index
                    end = yes_index
                if ex['answer'] == 'no' and tokens[start]!='no':
                    start = no_index
                    end = no_index
                
                _example  = {'tokens': tokens, 'answer':tokens[start : end + 1],'actual_answer':ex['answer'] ,'input_tokens':input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids, 'start':start, 'end':end}
                self.chunked_examples.append(_example)
    def __len__(self):
        return len(self.chunked_examples)

    def __getitem__(self, idx):
        return self.chunked_examples[idx]

class CustomDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.state = 0
        self.batch_state = 0
        self.examples = [i for i in range(len(self.dataset))]
        self.current_view = []
    
    def __len__(self):
        return math.ceil(len(self.examples)/self.batch_size)

    def prepare(self):
        shuffle(self.examples)
        self.state = 0
        self.batch_state = 0

    def restore(self, examples, state, batch_state):
        self.examples = examples
        self.state = state
        self.batch_state = batch_state
    
    def get(self):
        data_view = []
        for i in range(self.batch_size):
            if self.state + i < len(self.examples):
                data_view.append(self.dataset[self.examples[self.state + i]])
        self.state += self.batch_size
        self.batch_state+=1
        return data_view


if __name__=='__main__':
    from transformers import *
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CoQADataset('coqa.train.json')

    

