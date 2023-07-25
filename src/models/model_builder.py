from torch import arange
from transformers import BertModel

import torch
import torch.nn as nn

from neural_models import LayerNormLSTM

class Bert(nn.Module):
    def __init__(self, dr_rate):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, token_idx, attention_mask):

        hidden_staes, pooler= self.model(input_ids=token_idx, attention_mask=attention_mask.float().to(token_idx.device),
                              return_dict=False)

        output = hidden_staes[:, 0, :]


        return self.dropout(hidden_staes)

class Classifier(nn.Module):
    def __init__(self, hidden_size=768):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear(x).squeeze(-1)
        # h = self.linear1(x)
        sent_scores = self.sigmoid(h) * mask_cls.float()

        return sent_scores

class RNNEncoder(nn.Module):

    def __init__(self, bidirectional=True, num_layers=12, input_size=768,
                 hidden_size=768, dropout=0.0):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0

        hidden_size = self.hidden_size // num_directions

        self.rnn = LayerNormLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional)

        # self.rnn = nn.LSTM(
        #   input_size=self.input_size,
        #   hidden_size=hidden_size,
        #   num_layers=self.num_layers,
        #   bidirectional=bidirectional
        #   )

        self.wo = nn.Linear(num_directions * hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        x = torch.transpose(x, 1, 0)
        memory_bank, _ = self.rnn(x)
        memory_bank = self.dropout(memory_bank) + x
        memory_bank = torch.transpose(memory_bank, 1, 0)

        sent_scores = self.sigmoid(self.wo(memory_bank))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        return sent_scores


class Summarizer(nn.Module):
    def __init__(self, argument_train):
        super(Summarizer, self).__init__()

        self.bert = Bert(argument_train.drop_rate_bert)

        if(argument_train.encoder == "rnn"):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=argument_train.num_layers,
                                          input_size=argument_train.input_size, hidden_size=argument_train.hidden_size,
                                          dropout=argument_train.drop_rate_encoder)

        elif(argument_train.encoder == "rnn"):
            self.encoder = Classifier(argument_train.input_size)

    def forward(self, token_idx, attn_mask, cls_idx, cls_mask):

        top_vec = self.bert(token_idx, attn_mask)
        sents_vec = top_vec[arange(top_vec.size(0)).unsqueeze(1), cls_idx]

        sents_vec = sents_vec.detach().numpy() * cls_mask[:, :, None].float().detach().numpy()
        # sents_vec = sents_vec * cls_mask[:, :, None].float() #use if graphic card exists

        sent_scores = self.encoder(torch.from_numpy(sents_vec), cls_mask).squeeze(-1)
        # sent_scores = self.encoder(sents_vec, cls_mask).squeeze(-1)#use if graphic card exists

        return sent_scores, cls_mask


