
import pandas as pd
import glob
import json
from tqdm import tqdm


#for dataset from aihub & korean gov org of korean lang
def file_kor(file_atcl, file_labl):
    article = []
    tagged_id = []

    for idx, itm in enumerate(file_atcl['document']):
        for jtm in tqdm(itm):
            stnc_atcl = [ktm['form'] for ktm in jtm['paragraph']]
            article.append(stnc_atcl)

            for ktm in file_labl['data']:
                for ltm in ktm:
                    if(ltm['document_id'] == jtm['id']):
                        tagged_id.append([jtm['paragraph'].index(mtm["topic_sentences"]) for mtm in ltm])
                        break
                break
    return pd.DataFrame({'atcl': article, 'tgid': tagged_id})

def file_aihub(file):
    article = []
    tagged_id = []
    for itm in list(file['documents']):
        for jtm in itm:
            if(len(jtm['text']) < 2):
                article.append([ktm['sentence'] for ktm in jtm['text'][0]] )
                tagged_id.append(jtm['extractive'])
            else:
                if(isinstance(jtm['extractive'][0], list)):
                    article = article[:len(tagged_id)]
                    continue
                else:
                    if(None in jtm['extractive']):
                        article = article[:len(tagged_id)]
                        continue
                    article.append([ltm['sentence'] for ktm in jtm['text'] for ltm in ktm])
                    tagged_id.append(jtm['extractive'])

    return pd.DataFrame({'atcl': article, 'tgid': tagged_id})

def file_conc(dir):
    file_list = glob.glob(dir.replace("/", "\\") + '/*.json')

    df_list = []

    for itm in file_list:
        with open(itm, "rb") as f:
            df_list.append(pd.json_normalize(json.loads(f.read())))

    conc = pd.concat(df_list)

    return conc

