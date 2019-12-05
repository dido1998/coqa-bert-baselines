import torch
import torch.nn as nn
from utils.eval_utils import compute_eval_metric
from transformers import *
import numpy as np



class Model(nn.Module):
	def __init__(self, config, model, device, tokenizer):
		super(Model, self).__init__()
		self.device = device
		self.config = config

		if config['model_path'] is not None:
			self.pretrained_model = model[0].from_pretrained(config['model_path']).to(device)
		else:
			self.pretrained_model = model[0].from_pretrained(model[2]).to(device)
		self.pretrained_model.resize_token_embeddings(len(tokenizer))
		self.pretrained_model.train()
		self.qa_outputs = nn.Linear(768, 2)
		
	def forward(self, inputs, train = True):
		input_ids = torch.tensor([inp['input_tokens'] for inp in inputs], dtype = torch.long).to(self.device)
		input_mask = torch.tensor([inp['input_mask'] for inp in inputs], dtype = torch.long).to(self.device)
		segment_ids = torch.tensor([inp['segment_ids'] for inp in inputs], dtype = torch.long).to(self.device)
		start_positions = torch.tensor([inp['start'] for inp in inputs], dtype = torch.long).to(self.device)
		end_positions = torch.tensor([inp['end'] for inp in inputs], dtype = torch.long).to(self.device)

		if self.config['model_name'] == 'BERT' or self.config['model_name']=='SpanBERT':
			outputs = self.pretrained_model(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
		else:
			outputs = self.pretrained_model(input_ids, attention_mask = input_mask) # DistilBERT and RoBERTa do not use segment_ids 
		sequence_output = outputs[0]
		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)
		results = {}
		results['start_logits'] = start_logits
		results['end_logits'] = end_logits
		if train:
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			loss_fct = nn.CrossEntropyLoss()
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			total_loss = (start_loss + end_loss) / 2
			results['loss'] = total_loss
		return results

	def update(self, loss, optimizer, step):
		loss = loss.mean()
		
		loss.backward()
		
		
		if (step + 1) % self.config['gradient_accumulation_steps']:
			grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.config['grad_clip'])
			optimizer.step()
			optimizer.zero_grad()

	def evaluate(self, score_s, score_e, paragraphs, answers, debug = False):
	    if score_s.size(0) > 1:
	        score_s = score_s.exp().squeeze()
	        score_e = score_e.exp().squeeze()
	    else:
	        score_s = score_s.exp()
	        score_e = score_e.exp()
	    predictions = []
	    spans = []
	    for i, (_s, _e) in enumerate(zip(score_s, score_e)):
	        _s = _s.view(1, -1)
	        _e = _e.view(1, -1)
	        prediction, span = self._scores_to_text(paragraphs[i], _s, _e)
	        predictions.append(prediction)
	        spans.append(span)
	    answers = [[' '.join(a)] for a in answers]
	    f1, em = self.evaluate_predictions(predictions, answers)
	    return f1, em

	def _scores_to_text(self, text, score_s, score_e):
	    max_len = score_s.size(1)
	    scores = torch.ger(score_s.squeeze(), score_e.squeeze())
	    scores.triu_().tril_(max_len - 1)
	    scores = scores.cpu().detach().numpy()
	    s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
	    return ' '.join(text[s_idx: e_idx + 1]), (int(s_idx), int(e_idx))
	def evaluate_predictions(self,predictions, answers):
	    f1_score = compute_eval_metric('f1', predictions, answers)
	    em_score = compute_eval_metric('em', predictions, answers)
	    return f1_score, em_score