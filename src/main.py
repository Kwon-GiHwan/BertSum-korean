#python 3.10

#requirements
# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# pip install git+https://git@github.com/monologg/KoBERT-Transformers.git@master
#pip install sentencepiece
#pip install transformers
#pip install torch

from models import data_loader, model_builder, trainer
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

    df = pd.read_pickle(arg_train.dir_name + arg_train.train_file)
    df = df[['atcl', 'tgid']].iloc[:100]

    dset_train, dset_test = tts(df, test_size=0.1, random_state=42)

    dset_train = data_loader.BERTDataset(dset_train, train=True)
    dset_test = data_loader.BERTDataset(dset_test, train=True)

    dset_train = torch.utils.data.DataLoader(dset_train, batch_size=arg_train.batch_size, num_workers=1)
    dset_test = torch.utils.data.DataLoader(dset_test, batch_size=arg_train.batch_size, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_builder.Summarizer().to(device)

    trainer = trainer.Trainer(model, device, dset_train, dset_test, arg_train)

    epoch = arg_train.epoch

    for e in range(epoch):
        trainer.train()
        trainer.test()
        torch.save(model, arg_train.dir_name + arg_train.chk_point)


    #use ext_sentence here
    ext_sentence = ext_sentence.ExtSentence()

    df_sm = ext_sentence.refine_text(df_sm, 'atcl')
