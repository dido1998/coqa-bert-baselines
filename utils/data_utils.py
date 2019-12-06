from sogou_mrc.dataset.coqa import CoQAReader,CoQAEvaluator
from sogou_mrc.libraries.BertWrapper import BertDataHelper
from sogou_mrc.model.bert_coqa import BertCoQA
from sogou_mrc.data.vocabulary import  Vocabulary
import logging
import sys


def prepare_datasets(config, data = None, data_converted = None, prev_state):
    train_dataset = CoQADataset(config['trainset'], config['batch_size'], config['n_history'], True, data, data_converted, prev_state)
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

class CoQADataset(Dataset):
    """CoQA dataset."""

    def __init__(self, filename, batch_size, history_size, train = True, data = None, data_converted = None, prev_state = 0):
        #timer = Timer('Load %s' % filename)
        self.data_converted = data_converted
        self.data = data
        vocab = None
        self.prev_state = prev_state
        if data == None:
            data_reader = CoQAReader(history_size)
            vocab = Vocabulary()
            self.data = self.data_reader.read(filename, 'train' if train else 'dev')
            bert_dir = 'uncased_L-12_H-768_A-12'
            bert_data_helper = BertDataHelper(bert_dir)
            train_data = bert_data_helper.convert(train_data,data='coqa')
            self.data_converted = bert_data_helper.convert(train_data, data = 'coqa') 
        else:
            vocab = Vocabulary()
            vocab.build_vocab(self.data)

        self.batches = BatchGenerator(vocab, self.data_converted,training=train,batch_size=batch_size,
            additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position', 'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'],
            shuffle= True if data == None else False)
        c = 0
        while c < self.prev_state:
            self.batches.next()
        def init(self):
            self.batches.init()
        
        def __len__(self):
            return self.batches.instance_size()
        
        def next(self):
            self.prev_state+=1
            return self.batches.next()

    

