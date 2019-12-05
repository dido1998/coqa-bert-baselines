import torch
import torch.nn as nn
from utils.data_utils import prepare_datasets
from utils.eval_utils import AverageMeter
from model import Model
from transformers import *
import time
import os

MODELS = {'BERT':(BertModel,       BertTokenizer,       'bert-base-uncased'),
          'DistilBERT':(DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          'RoBERTa':(RobertaModel,    RobertaTokenizer,    'roberta-base'),
          'SpanBERT':(BertModel, BertTokenizer, 'bert-base-cased')}



class ModelHandler():
	def __init__(self, config):
		self.config = config
		tokenizer_model = MODELS[config['model_name']]
		self.train_loader, self.dev_loader, tokenizer = prepare_datasets(config, tokenizer_model)
		self._n_dev_batches = len(self.dev_loader.dataset) // config['batch_size']
		self._n_train_batches = len(self.train_loader.dataset) // config['batch_size']
		if config['cuda']:
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self._train_loss = AverageMeter()
		self._train_f1 = AverageMeter()
		self._train_em = AverageMeter()
		self._dev_f1 = AverageMeter()
		self._dev_em = AverageMeter()

		self.model = Model(config, MODELS[config['model_name']], self.device, tokenizer).to(self.device)
		t_total = len(self.train_loader) // config['gradient_accumulation_steps'] * config['max_epochs']
		self.optimizer = AdamW(self.model.parameters(), lr=config['lr'], eps = config['adam_epsilon'] )
		self.optimizer.zero_grad()
		self._n_train_examples = 0
		self._epoch = self._best_epoch = 0
		self._best_f1 = 0
		self._best_em = 0
		self.restored = False
		if config['pretrained_dir'] is not None:
			self.restore()

	def train(self):
		if not self.restored:
			print("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
			self._run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'], save = False)
			
			format_str = "Validation Epoch {} -- F1: {:0.2f}, EM: {:0.2f} --"
			print(format_str.format(self._epoch, self._dev_f1.mean(), self._dev_em.mean()))
			self._best_f1 = self._dev_f1.mean()
			self._best_em = self._dev_em.mean()
		while self._stop_condition(self._epoch):
			self._epoch += 1
			print("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
			if not self.restored:
				self.train_loader.prepare()	 
			self.restored = False

			self._run_epoch(self.train_loader, training=True, verbose=self.config['verbose'])
			format_str = "Training Epoch {} -- Loss: {:0.4f}, F1: {:0.2f}, EM: {:0.2f} --"
			print(format_str.format(self._epoch, self._train_loss.mean(),
			self._train_f1.mean(), self._train_em.mean()))
			print("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
			self.dev_loader.prepare()
			self._run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'], save = False)
			format_str = "Validation Epoch {} -- F1: {:0.2f}, EM: {:0.2f} --"
			print(format_str.format(self._epoch, self._dev_f1.mean(), self._dev_em.mean()))
			
			if self._best_f1 <= self._dev_f1.mean():
			    self._best_epoch = self._epoch
			    self._best_f1 = self._dev_f1.mean()
			    self._best_em = self._dev_em.mean()
			    print("!!! Updated: F1: {:0.2f}, EM: {:0.2f}".format(self._best_f1, self._best_em))
			self._reset_metrics()
			self.save(self._epoch)

	def restore(self):
		if not os.path.exists(self.config['pretrained_dir']):
			print('dir doesn\'t exists, cannot restore')
			return
		restored_params = torch.load(self.config['pretrained_dir']+'/latest/model.pth')
		self.model.load_state_dict(restored_params['model'])
		self.optimizer.load_state_dict(restored_params['optimizer'])
		self._epoch = restored_params['epoch']
		self._best_epoch = restored_params['best_epoch']
		self._n_train_examples = restored_params['train_examples']
		self._best_f1 = restored_params['best_f1']
		self._best_em = restored_params['best_em']
		examples = restored_params['dataloader_examples']
		batch_state = restored_params['dataloader_batch_state']
		state = restored_params['dataloader_state']
		self.train_loader.restore(examples, state, batch_state)

		self.restored = True

	def save(self, save_epoch_val):
		if not os.path.exists(self.config['save_state_dir']):
			os.mkdir(self.config['save_state_dir'])
		
		if self._best_epoch == self._epoch:
			if not os.path.exists(self.config['save_state_dir']+'/best'):
				os.mkdir(self.config['save_state_dir']+'/best')
			save_dic = {'epoch':self._epoch,
			'best_epoch': self._best_epoch,
			'train_examples':self._n_train_examples,
			'model':self.model.state_dict(),
			'optimizer':self.optimizer.state_dict(),
			'best_f1':self._best_f1,
			'best_em':self._best_em}
			torch.save(save_dic, self.config['save_state_dir']+'/best/model.pth')
		if not os.path.exists(self.config['save_state_dir']+'/latest'):
			os.mkdir(self.config['save_state_dir']+'/latest')
		save_dic = {'epoch':save_epoch_val,
			'best_epoch': self._best_epoch,
			'train_examples':self._n_train_examples,
			'model':self.model.state_dict(),
			'optimizer':self.optimizer.state_dict(),
			'best_f1':self._best_f1,
			'best_em':self._best_em,
			'dataloader_batch_state': self.train_loader.batch_state,
			'dataloader_state':self.train_loader.state,
			'dataloader_examples':self.train_loader.examples}
		torch.save(save_dic, self.config['save_state_dir']+'/latest/model.pth')


	def _run_epoch(self, data_loader, training=True, verbose=10, out_predictions=False, save = True):
	    start_time = time.time()
	    while data_loader.batch_state < len(data_loader):
	        input_batch = data_loader.get()
	        res = self.model(input_batch, training)
	        tr_loss = 0
	        if training:
	        	loss = res['loss']
	        	if self.config['gradient_accumulation_steps'] > 1:
	        		loss = loss / self.config['gradient_accumulation_steps']
	        	tr_loss = loss.mean().item()
	        start_logits = res['start_logits']
	        end_logits = res['end_logits']
	        
	        if training:
	        	self.model.update(loss, self.optimizer, data_loader.batch_state)
	        paragraphs = [inp['tokens'] for inp in input_batch]
	        answers = [inp['answer'] for inp in input_batch]
	        f1, em = self.model.evaluate(start_logits, end_logits, paragraphs, answers)

	        self._update_metrics(tr_loss, f1, em, len(paragraphs), training=training)

	        if training:
	            self._n_train_examples += len(paragraphs)
	        if (verbose > 0) and (data_loader.batch_state % verbose == 0):
	            if save:
	            	self.save(self._epoch - 1)
	            mode = "train" if training else "dev"
	            print(self.report(data_loader.batch_state, tr_loss, f1 * 100, em * 100, mode))
	            print('used_time: {:0.2f}s'.format(time.time() - start_time))

	def _update_metrics(self, loss, f1, em, batch_size, training=True):
		if training:
			self._train_loss.update(loss)
			self._train_f1.update(f1 * 100, batch_size)
			self._train_em.update(em * 100, batch_size)
		else:
			self._dev_f1.update(f1 * 100, batch_size)
			self._dev_em.update(em * 100, batch_size)

	def _reset_metrics(self):
		self._train_loss.reset()
		self._train_f1.reset()
		self._train_em.reset()
		self._dev_f1.reset()
		self._dev_em.reset()
	def report(self, step, loss, f1, em, mode='train'):
		if mode == "train":
		    format_str = "[train-{}] step: [{} / {}] | exs = {} | loss = {:0.4f} | f1 = {:0.2f} | em = {:0.2f}"
		    return format_str.format(self._epoch, step, self._n_train_batches, self._n_train_examples, loss, f1, em)
		elif mode == "dev":
		    return "[predict-{}] step: [{} / {}] | f1 = {:0.2f} | em = {:0.2f}".format(
		            self._epoch, step, self._n_dev_batches, f1, em)
		elif mode == "test":
		    return "[test] | test_exs = {} | step: [{} / {}] | f1 = {:0.2f} | em = {:0.2f}".format(
		            self._n_test_examples, step, self._n_test_batches, f1, em)
		else:
			raise ValueError('mode = {} not supported.' % mode)
	def _stop_condition(self, epoch):
		"""
		Checks have not exceeded max epochs and has not gone 10 epochs without improvement.
		"""
		no_improvement = epoch >= self._best_epoch + 10
		exceeded_max_epochs = epoch >= self.config['max_epochs']
		return False if exceeded_max_epochs or no_improvement else True