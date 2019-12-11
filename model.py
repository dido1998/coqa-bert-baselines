import torch
import torch.nn as nn
from utils.eval_utils import compute_eval_metric
from transformers import *
import numpy as np
from collections import OrderedDict

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def one_hot(a, num_classes, device):
	b = torch.zeros(a.size(0), num_classes)
	for i in range(a.size(0)):
		b[i, a[i]] = 1
	return b.to(device)

VERY_NEGATIVE_NUMBER = -1e29

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
		self.yes_output = nn.Linear(768, 1)
		self.no_output = nn.Linear(768, 1)
		self.unk_output = nn.Linear(768, 1)

		
	def forward(self, inp, train = True):
		input_ids = torch.tensor(inp['input_ids'], dtype = torch.long).to(self.device)
		input_mask = torch.tensor(inp['input_mask'], dtype = torch.long).to(self.device)
		segment_ids = torch.tensor(inp['segment_ids'], dtype = torch.long).to(self.device)
		start_positions = torch.tensor(inp['start_position'], dtype = torch.long).to(self.device)
		end_positions = torch.tensor(inp['end_position'], dtype = torch.long).to(self.device)
		yes_mask = torch.tensor(inp['yes_mask'], dtype = torch.float32).to(self.device).view(-1, 1)
		no_mask = torch.tensor(inp['no_mask'], dtype = torch.float32).to(self.device).view(-1, 1)
		unk_mask = torch.tensor(inp['unk_mask'], dtype = torch.float32).to(self.device).view(-1, 1)
		extractive_mask = torch.tensor(inp['extractive_mask'], dtype = torch.float32).to(self.device).view(-1,1)

		if self.config['model_name'] == 'BERT' or self.config['model_name']=='SpanBERT':
			outputs = self.pretrained_model(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
		else:
			outputs = self.pretrained_model(input_ids, attention_mask = input_mask) # DistilBERT and RoBERTa do not use segment_ids 
		sequence_output = outputs[0]
		pooled_output = outputs[1]
		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)
		yes_logit = self.yes_output(pooled_output).view(-1, 1)
		no_logit = self.no_output(pooled_output).view(-1, 1)
		unk_logit = self.unk_output(pooled_output).view(-1, 1)
		results = {}
		
		if train:
			#print(start_logits.size())
			#print(input_mask.size())
			masked_start_logits = start_logits * input_mask + (1 - input_mask) * VERY_NEGATIVE_NUMBER
			masked_end_logits = end_logits * input_mask + (1 - input_mask) * VERY_NEGATIVE_NUMBER
			start_logits_  = torch.cat([masked_start_logits, yes_logit, no_logit, unk_logit], dim = 1)
			end_logits_ = torch.cat([masked_end_logits, yes_logit, no_logit, unk_logit], dim = 1)
			start_mask = one_hot(start_positions, start_logits.size(1), self.device)
			end_mask = one_hot(end_positions, end_logits.size(1), self.device)
			start_mask = start_mask * extractive_mask
			end_mask = end_mask * extractive_mask
			start_mask_ = torch.cat([start_mask, yes_mask, no_mask, unk_mask], dim = 1)
			end_mask_ = torch.cat([end_mask, yes_mask, no_mask, unk_mask], dim = 1)
			_, start_targets = start_mask.max(dim= 1)
			_, end_targets = end_mask.max(dim = 1)
			loss_fct = nn.CrossEntropyLoss()
			start_loss = loss_fct(start_logits_, start_targets)
			end_loss = loss_fct(end_logits_, end_targets)
			total_loss = (start_loss + end_loss) / 2
			results['loss'] = total_loss
		results['output'] = OrderedDict({
			'start_logits': start_logits.cpu().data.numpy(),
			'end_logits': end_logits.cpu().data.numpy(),
			'unk_logits': unk_logit.cpu().data.numpy(),
			'yes_logits': yes_logit.cpu().data.numpy(),
			'no_logits':no_logit.cpu().data.numpy()
			})
			
		return results

	def update(self, loss, optimizer, step):
		loss = loss.mean()
		
		loss.backward()
		
		
		if (step + 1) % self.config['gradient_accumulation_steps']:
			grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.config['grad_clip'])
			optimizer.step()
			optimizer.zero_grad()

	def prediction_to_ori(self, start_index, end_index, instance):
	    if start_index > 0:
	        tok_tokens = instance['tokens'][start_index:end_index + 1]
	        orig_doc_start = instance['token_to_orig_map'][start_index]
	        orig_doc_end = instance['token_to_orig_map'][end_index]
	        char_start_position = instance["context_token_spans"][orig_doc_start][0]
	        char_end_position = instance["context_token_spans"][orig_doc_end][1]
	        pred_answer = instance["context"][char_start_position:char_end_position]
	        return pred_answer
	    return ""


	def get_best_answer(self, output, instances, max_answer_len=11, null_score_diff_threshold=0.0):
	    def _get_best_indexes(logits, n_best_size):
	        """Get the n-best logits from a list."""
	        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

	        best_indexes = []
	        for i in range(len(index_and_score)):
	            if i >= n_best_size:
	                break
	            best_indexes.append(index_and_score[i][0])
	        return best_indexes

	    ground_answers = []
	    qid_with_max_logits = {}
	    qid_with_final_text = {}
	    qid_with_no_logits = {}
	    qid_with_yes_logits = {}
	    qid_with_unk_logits = {}
	    #print(len(instances))
	    #print(output['start_logits'].shape)
	    for i in range(len(instances)):
	        instance = instances[i]
	        ground_answers.append(instance['answer'])
	        start_logits = output['start_logits'][i]
	        end_logits = output['end_logits'][i]
	        feature_unk_score = output['unk_logits'][i][0] * 2
	        feature_yes_score = output['yes_logits'][i][0] * 2
	        feature_no_score = output['no_logits'][i][0] * 2
	        start_indexes = _get_best_indexes(start_logits, n_best_size=20)
	        end_indexes = _get_best_indexes(end_logits, n_best_size=20)
	        max_start_index = -1
	        max_end_index = -1
	        max_logits = -100000000
	        for start_index in start_indexes:
	            for end_index in end_indexes:
	                if start_index >= len(instance['tokens']):
	                    continue
	                if end_index >= len(instance['tokens']):
	                    continue
	                if start_index not in instance['token_to_orig_map']:
	                    continue
	                if end_index not in instance['token_to_orig_map']:
	                    continue
	                if end_index < start_index:
	                    continue
	                if not instance['token_is_max_context'].get(start_index, False):
	                    continue
	                length = end_index - start_index - 1
	                if length > max_answer_len:
	                    continue
	                sum_logits = start_logits[start_index] + end_logits[end_index]
	                if sum_logits > max_logits:
	                    max_logits = sum_logits
	                    max_start_index = start_index
	                    max_end_index = end_index
	        final_text = ''
	        if (max_start_index != -1 and max_end_index != -1):
	            final_text = self.prediction_to_ori(max_start_index, max_end_index, instance)
	        story_id, turn_id = instance["qid"].split("|")
	        turn_id = int(turn_id)
	        if (story_id, turn_id) in qid_with_max_logits and max_logits > qid_with_max_logits[(story_id, turn_id)]:
	            qid_with_max_logits[(story_id, turn_id)] = max_logits
	            qid_with_final_text[(story_id, turn_id)] = final_text
	        if (story_id, turn_id) not in qid_with_max_logits:
	            qid_with_max_logits[(story_id, turn_id)] = max_logits
	            qid_with_final_text[(story_id, turn_id)] = final_text
	        if (story_id, turn_id) not in qid_with_no_logits:
	            qid_with_no_logits[(story_id, turn_id)] = feature_no_score
	        if feature_no_score > qid_with_no_logits[(story_id, turn_id)]:
	            qid_with_no_logits[(story_id, turn_id)] = feature_no_score
	        if (story_id, turn_id) not in qid_with_yes_logits:
	            qid_with_yes_logits[(story_id, turn_id)] = feature_yes_score
	        if feature_yes_score > qid_with_yes_logits[(story_id, turn_id)]:
	            qid_with_yes_logits[(story_id, turn_id)] = feature_yes_score
	        if (story_id, turn_id) not in qid_with_unk_logits:
	            qid_with_unk_logits[(story_id, turn_id)] = feature_unk_score
	        if feature_unk_score > qid_with_unk_logits[(story_id, turn_id)]:
	            qid_with_unk_logits[(story_id, turn_id)] = feature_unk_score
	    result = {}
	    for k in qid_with_max_logits:
	        scores = [qid_with_max_logits[k], qid_with_no_logits[k], qid_with_yes_logits[k], qid_with_unk_logits[k]]
	        max_val = max(scores)
	        if max_val == qid_with_max_logits[k]:
	            result[k] = qid_with_final_text[k]
	        elif max_val == qid_with_unk_logits[k]:
	            result[k] = 'unknown'
	        elif max_val == qid_with_yes_logits[k]:
	            result[k] = 'yes'
	        else:
	            result[k] = 'no'
	    return result
 

	def evaluate(self, evaluator, output, instances):
	    result = self.get_best_answer(output, instances)
	    #print(result)
	    score = evaluator.get_score(result)
	    return score['f1'], score['em']

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