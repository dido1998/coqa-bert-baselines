from data_helper.coqa import CoQAReader
from data_helper.BertWrapper import BertDataHelper
from data_helper.bert_coqa import BertCoQA
from data_helper.vocabulary import  Vocabulary
from data_helper.batch_generator import BatchGenerator
from random import shuffle
import logging
import sys


def prepare_datasets(config, data = None, prev_state = 0):
    train_dataset = CoQADataset(config['trainset'], config['batch_size'], config['n_history'], True, data, prev_state)
    eval_dataset = CoQADataset(config['devset'], config['batch_size'], config['n_history'], False)
    return train_dataset, eval_dataset





def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)

class CoQADataset():
    """CoQA dataset."""

    def __init__(self, filename, batch_size, history_size, train = True, data = None, prev_state = 0):
        #timer = Timer('Load %s' % filename)
        self.data = data
        self.vocab = None
        self.prev_state = prev_state
        self.batch_size = batch_size
        self.train = train
        bert_dir = '/home/aniket/coqa/uncased_L-12_H-768_A-12'
        self.bert_data_helper = BertDataHelper(bert_dir)
        if data == None:
            data_reader = CoQAReader(history_size)
            self.vocab = Vocabulary()
            self.data = data_reader.read(filename, 'train' if train else 'dev')
            self.vocab.build_vocab(self.data)
            shuffle(self.data)
        else:
            self.vocab = Vocabulary()
            self.vocab.build_vocab(self.data)

        #self.batches = BatchGenerator(vocab, self.data_converted,training=train,batch_size=batch_size,
        #    additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position'],#, 'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'],
        #    shuffle= True if data == None else False)
        
    def __len__(self):
        return len(self.data)
        
    def next(self):
        data_converted = self.bert_data_helper.convert(self.data[self.prev_state:self.prev_state + self.batch_size])
        batch = BatchGenerator(self.vocab, data_converted, training = self.train, batch_size = self.batch_size, additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position'],#, 'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'],
        shuffle=False)
        self.prev_state+=self.batch_size
        batch.init()
        return batch.next()

    

if __name__=='__main__':
    dataset = CoQADataset('/home/aniket/coqa/data/coqa-train-v1.0.json',4,2)
    print(len(dataset))