import torch
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup as cosine_warmup
from torch import nn
from tqdm import tqdm

import numpy as np

import math

class Trainer(nn.Module):
    def __init__(self, model, device, dset_train, dset_test, arg_train = None):
        super(Trainer, self).__init__()

        self.batch_size = arg_train.batch_size
        self.warmup_rate = arg_train.warmup_rate
        self.epoch = arg_train.epoch
        self.grad_norm = arg_train.grad_norm
        self.learn_rate = arg_train.learn_rate

        self.dset_train = dset_train
        self.dset_test = dset_test

        self.device = device
        self.model = model


        self.no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_param = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in self.no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in self.no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(self.optimizer_param, lr=self.learn_rate)
        self.loss_fn = nn.BCELoss(reduction=None)

        self.train_len = len(self.dset_train) * self.epoch
        self.warmup_step = int(self.train_len * self.warmup_rate)

        self.scheduler = cosine_warmup(self.optimizer, num_warmup_steps=self.warmup_step, num_training_steps=self.train_len)

    def calc_acc(self, sent_score, label):
        slct_idx = np.argsort(-sent_score, 1)
        label = label.numpy()

        acc = 0.0
        for idx, itm in enumerate(slct_idx):
            post_idx = np.where(label[idx] == 1)
            res = np.in1d(itm, post_idx)
            acc += (res == True).sum()

        return 100 * (acc / len(label))


    def rdass(self, input_idx, label, sent_score):#seems not useful to extract summarization task


        pass

    def train_loop(self):

        self.model.train()
        train_acc = 0.0

        for batch_id, (_input_idx, _attention_mask, _cls_idx, _cls_mask, _label) in enumerate(tqdm(self.dset_train)):
            self.optmzr.zero_grad()

            input_idx = _input_idx.to(self.device)
            attention_mask = _attention_mask.to(self.device)
            cls_idx = cls_idx.to(self.device)
            cls_mask = cls_mask.to(self.device)
            label = _label.to(self.device)

            sent_score, cls_mask = self.model(input_idx, attention_mask, cls_idx, cls_mask)

            loss = self.loss_fn(sent_score, label.float())
            loss = (loss * cls_mask.float()).sum()
            loss.requires_grad_(True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)  # gradient clipping
            self.optmzr.step()
            self.scheduler.step()

            train_acc += self.calc_acc(sent_score, label.float())

        print(
            "train loop: loss {}  acc {}".format(loss.data.cpu().numpy(), train_acc / len(self.dset_train)))

    def test_loop(self):

        self.model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_acc = 0.0

            for batch_id, (input_idx, attention_mask, _cls_idx, _cls_mask, _label) in enumerate(tqdm(self.dset_test)):
                input_idx = input_idx.to(self.device)
                attention_mask = attention_mask.to(self.device)
                cls_idx = _cls_idx.to(self.device)
                cls_mask = _cls_mask.to(self.device)
                label = _label.to(self.device)

                sent_score, cls_mask = self.model(input_idx, attention_mask, cls_idx, cls_mask)

                test_loss = self.loss_fn(sent_score, label.float())
                test_loss = (test_loss * cls_mask.float()).sum()

                sent_score = sent_score + cls_mask.float()
                sent_score = sent_score.cpu().data.numpy()

                test_acc += self.calc_acc(sent_score, label.float())

            print(
                "test loop: loss {}  acc {}".format( test_loss.data.cpu().numpy(), test_acc / len(self.dset_test)))

class Validator(nn.Module):
    def __init__(self, model, device, tokenizer,dset_summ, arg_valid = None):
        super(Trainer, self).__init__()

        self.batch_size = arg_valid.batch_size

        self.dset_summ = dset_summ

        self.device = device
        self.model = model
        self.tokenizer = tokenizer
    def summarize_task(self):

        self.model.eval()
        with torch.no_grad():
            pred_atcl = []

            for batch_id, (input_idx, attention_mask, _cls_idx, _cls_mask) in enumerate(tqdm(self.dset_summ)):
                input_idx = input_idx.to(self.device)
                attention_mask = attention_mask.to(self.device)
                cls_idx = _cls_idx.to(self.device)
                cls_mask = _cls_mask.to(self.device)

                sent_score, cls_mask = self.model(input_idx, attention_mask, cls_idx, cls_mask)

                sent_score = sent_score + cls_mask.float()
                sent_score = sent_score.cpu().data.numpy()

                slct_idx = np.argsort(-sent_score, 1)

                for idx, itm in enumerate(slct_idx):
                    pred_stnc = []
                    src_stnc = self.tokenizer.decode(input_idx[idx]).split('[SEP]')

                    for jtm in itm[:len(src_stnc)]:
                        try:
                            candid = src_stnc[jtm]
                            pred_stnc.append(candid)
                        except:
                            pass

                    rel_len = math.ceil(len(src_stnc) * 0.3)

                    pred_stnc = ''.join(pred_stnc[:rel_len])
                    pred_stnc = pred_stnc.replace('[CLS]', '')
                    pred_stnc = pred_stnc.replace('[SEP]', '')
                    pred_stnc = pred_stnc.replace('[UNK]', '')
                    pred_stnc = pred_stnc.replace('[PAD]', '')

                    pred_atcl.append(pred_stnc)

            return pred_atcl
