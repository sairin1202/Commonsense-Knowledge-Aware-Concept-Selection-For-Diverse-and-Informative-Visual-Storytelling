from collections import namedtuple
import random
from tqdm import tqdm, trange
import os
import nltk

import torch

from generator import SequenceGenerator

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam

from bart_utils import BARTModelWrapper


BART_MAX_LEN = 1024


TextPairData = namedtuple('TextPairData', ['src_text', 'tgt_text'])


class BART:
    def __init__(self, opt):
        self._model = BARTModelWrapper(opt)

        self._optimizer = None
        self._lr_scheduler = None
        self._global_step = 0

        self._dataset = {}
        self._log_dir = None
        self._eval_steps = None
        self._log_file = None
        self._best_dev_loss = None

    def create_training_log(self, eval_steps, label):
        self._log_dir = f'training_logs/{label}'
        self._eval_steps = eval_steps
        self._best_dev_loss = float('inf')

        os.makedirs(os.path.join(self._log_dir, 'ckpt_gens'), exist_ok=True)
        self._log_file = open(os.path.join(self._log_dir, 'log.txt'), 'w')

    def get_optimizer(self, lr, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.visual_encoder.named_parameters()]},]
        # self._optimizer = Adam(params = optimizer_grouped_parameters, lr = lr)
        self._optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

    def get_bart_optimizer(self, lr, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in self._model._interface.model.named_parameters()]},
        #     {"params": [p for n, p in self._model.visual_encoder.named_parameters()], "lr":1e-4}]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()]},]
        # self._optimizer = Adam(params = optimizer_grouped_parameters, lr = lr)
        self._optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print(f'Model saved in {path}.')

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path, map_location='cuda'))
        print(f'Model {path} loaded.')

    def load_data(self, set_type, src_texts, tgt_texts):
        assert len(src_texts) == len(tgt_texts)

        self._dataset[set_type] = []
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            self._dataset[set_type].append(TextPairData(
                src_text=src_text, tgt_text=tgt_text))

  


    def generate(self, cond, keys, top_k, top_p):
        self._model.split_to_gpus(1)
        self._model.eval()
        key_tokens = self._model.encode(keys)[:BART_MAX_LEN].unsqueeze(0)
        generator = SequenceGenerator(
            tgt_dict=self._model.dictionary,
            max_len_b=BART_MAX_LEN,
            sampling=True,
            sampling_topk=-1,
            sampling_topp=0.9)

        # src_tokens = self._model.encode(cond)[:BART_MAX_LEN]
        outputs = generator.generate(
            models=[self._model],
            sample={'net_input': {
                'src_feats': cond.to('cuda'),
                'keys': key_tokens.to('cuda'),
                'src_lengths': torch.tensor([cond.shape[1]+key_tokens.shape[1]])
            }})
        # words = []
        # for w in outputs[0][0]['tokens']:
        #     words.append(self._model.dictionary.string(w))
        return self._model.decode(outputs[0][0]['tokens'].cpu())

    def gen_log(self):
        eval_loss = self.evaluate()

        print(f'Global Step: {self._global_step}, Eval Loss: {eval_loss}',
              file=self._log_file)

        if eval_loss < self._best_dev_loss:
            self._best_dev_loss = eval_loss
            self.save_model(f'{self._log_dir}/best_model.pt')
            print('Best Model Updated.', file=self._log_file)

        self._log_file.flush()

        generation_file = open(
            f'{self._log_dir}/ckpt_gens/step{self._global_step}.txt', 'w')

        for example in self._dataset['dev'][:10]:
            gen_text = self.generate(example.src_text, top_k=-1., top_p=0.95)

            print('SOURCE:\n', example.src_text, '\n', '-' * 50, '\n',
                  'GENERATION:\n', gen_text, '\n', '-' * 50, '\n',
                  'TARGET:\n', example.tgt_text, '\n', '=' * 100, '\n\n\n',
                  file=generation_file)
            generation_file.flush()

    def _get_seq2seq_loss(self, src_feat, keys, tgt_text):

        key_tokens = self._model.encode(keys)[:BART_MAX_LEN].unsqueeze(0)
        tgt_tokens = self._model.encode(tgt_text)[:BART_MAX_LEN].unsqueeze(0)

        logits, extra = self._model(
            src_feat=src_feat,
            keys = key_tokens,
            src_lengths=torch.tensor([src_feat.shape[1]+key_tokens.shape[1]]),
            prev_output_tokens=tgt_tokens)
            
        key_tokens = key_tokens.to(logits.device)
        tgt_tokens = tgt_tokens.to(logits.device)
        # print(key_tokens)
        # print(tgt_tokens)
        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self._model.dictionary.pad())
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


    @property
    def dataset(self):
        return self._dataset

    @property
    def get_lr(self):
        return self._lr_scheduler.get_lr()
