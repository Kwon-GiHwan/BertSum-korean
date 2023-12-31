import torch
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm

class Processor():
    def __init__(self, tokenizer, tokenizer_len):

        self.tokenizer = tokenizer

        self.cls_token = self.tokenizer.convert_tokens_to_ids("[CLS]")

        self.tokenizer_len = tokenizer_len
    def padd(self, data, pad_id, width=-1, tensor=True):

        if (width == -1):
            width = max(len(d) for d in data)
        if(tensor):
            padded_data = [torch.IntTensor(d + [pad_id] * (width - len(d))) for d in data]
        else:
            padded_data = [d + [pad_id] * (width - len(d)) for d in data]
        return padded_data
    def mask_bitwise(self, masked_list, bit):
        for idx, itm in enumerate(masked_list):
            for jdx, jtm in enumerate(itm):
                if jtm == bit:
                    masked_list[idx][jdx] = 0
                else:
                    masked_list[idx][jdx] = 1

            masked_list[idx] = torch.IntTensor(masked_list[idx])

        return masked_list
    def sentence_label(self, article):

        text = " [SEP] [CLS] ".join(article)#재검토

        sentence = self.tokenizer(text, add_special_tokens=True, truncation=True, padding='max_length',
                                  max_length=self.tokenizer_len, return_attention_mask=True)

        input_idx = sentence['input_ids']
        attention_mask = sentence['attention_mask']

        cls_idx = [idx for idx, itm in enumerate(input_idx) if itm == self.cls_token]

        cls_mask = np.zeros(len(input_idx)).tolist()

        for itm in cls_idx:
            cls_mask[itm] = 1

        input_idx = torch.IntTensor(input_idx)
        attention_mask = torch.IntTensor(attention_mask)

        return [input_idx, attention_mask, cls_idx, cls_mask]

    def data_cls(self, article):

        len_max = max([len(i) for i in article])

        empty_lst = np.empty((len(article), len_max), dtype='object').tolist()
        dic_ret = {}

        dic_ret['inpt_idx'] = empty_lst
        dic_ret['attn_mask'] = empty_lst
        dic_ret['cls_idx'] = empty_lst
        dic_ret['cls_mask'] = empty_lst

        for idx, itm in tqdm(enumerate(article)):
            ret = self.sentence_label(itm)
            dic_ret['inpt_idx'][idx] = ret[0]
            dic_ret['attn_mask'][idx] = ret[1]
            dic_ret['cls_idx'][idx] = ret[2]
            dic_ret['cls_mask'][idx] = ret[3]

        inpt_idx = list(dic_ret['inpt_idx'])
        attn_mask = list(dic_ret['attn_mask'])
        cls_mask = self.mask_bitwise(self.padd(dic_ret['cls_idx'], -1, tensor=False), -1)
        cls_idx = self.padd(dic_ret['cls_idx'], 0)

        return (inpt_idx, attn_mask, cls_idx, cls_mask)

class ValidDataset(Dataset):
    def __init__(self, tokenizer, article, arg_valid):
        self.processor = Processor(tokenizer, arg_valid.tokenizer_len)
        self.dset = self.processor.data_cls(article)

    def __getitem__(self, idx):
        return (self.dset[0][idx], self.dset[1][idx], self.dset[2][idx],
                self.dset[3][idx])

    def __len__(self):
        return (len(self.dset[0]))
