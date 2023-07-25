#python 3.10

#requirements
# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# pip install git+https://git@github.com/monologg/KoBERT-Transformers.git@master
#pip install sentencepiece
#pip install transformers
#pip install torch

from models import data_loader, model_builder, trainer
from kobert_transformers.tokenization_kobert import KoBertTokenizer

from pre_process import data_parser
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import torch
import argparse
import json

def add_argument(file_loc):
    parser = argparse.ArgumentParser()

    try:
        config = json.load(open(file_loc, 'r'))
        t_args = argparse.Namespace()
        t_args.__dict__.update(config)
        args = parser.parse_args(namespace=t_args)

        return args

    except:

        parser.add_argument("-dir_name", default='../data/korean')
        parser.add_argument("-chk_point", default='../models/')

        parser.add_argument("-tokenizer_len", default=512)

        parser.add_argument("-encoder", default='classifier', type=str,
                            choices=['classifier', 'rnn'])
        parser.add_argument("-mode", default='train', type=str, choices=['train', 'summary'])

        parser.add_argument("-input_size", default=768)
        parser.add_argument("-hidden_size", default=768)
        parser.add_argument("-num_layer", default=2)

        parser.add_argument("-drop_rate_encoder", default=0.5)
        parser.add_argument("-drop_rate_bert", default=0.5)

        parser.add_argument("-batch_size", default=8)
        parser.add_argument("-warmup_rate", default=1)
        parser.add_argument("-epoch", default=10)
        parser.add_argument("-grad_norm", default=1)
        parser.add_argument("-learn_rate", default=1)


if __name__ == '__main__':
    arg_train = add_argument()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_builder.Summarizer().to(device)
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert',
                                                sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6,
                                                                 'enable_sampling': True})

    if(arg_train.mode == "train"):
        epoch = arg_train.epoch

        df = pd.read_pickle(arg_train.dir_name + arg_train.train_file)
        df = df[['atcl', 'tgid']].iloc[:100]

        dset_train, dset_test = tts(df, test_size=0.1, random_state=42)

        dset_train = data_loader.BERTDataset(tokenizer, dset_train, train=True)
        dset_test = data_loader.BERTDataset(tokenizer, dset_test, train=True)

        dset_train = torch.utils.data.DataLoader(dset_train, batch_size=arg_train.batch_size, num_workers=1)
        dset_test = torch.utils.data.DataLoader(dset_test, batch_size=arg_train.batch_size, num_workers=1)

        trainer = trainer.Trainer(model, device, dset_train, dset_test, arg_train)

        for e in range(epoch):
            trainer.train()
            trainer.test()
            torch.save(model, arg_train.dir_name + arg_train.chk_point)

    elif(arg_train.mode == "summarization"):

        df = pd.read_pickle(arg_train.dir_name + arg_train.train_file)
        df = df[['atcl', 'tgid']].iloc[:100]



        dset_summ = data_loader.BERTDataset(tokenizer, df, train=True)
        dset_summ = torch.utils.data.DataLoader(dset_summ, batch_size=arg_train.batch_size, num_workers=1)

        trainer = trainer.Validator(model, device, tokenizer, dset_summ, arg_train)

        ret = trainer.train()

        df['summ'] = ret

        pd.to_excel(arg_train.dir_name + 'result.xlsx')

