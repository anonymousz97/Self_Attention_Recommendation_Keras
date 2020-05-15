import pandas as pd
import numpy as np
from tqdm import tqdm

def split_session(url_file,max_len):
    list_session = []
    list_label = []
    df = pd.read_json(url_file)
    df = df.drop(df[df['videoids'].map(len) > 50].index)
    df = df.tail(10000)
    for i in tqdm(df['videoids']):
        for j in range(1,len(i)):
            list_t = []
            list_label.append(i[j])
            if j-max_len<0:
                list_t = i[:j]
            else:
                list_t = i[j-max_len:j]
            list_session.append(list_t)
    for i in list_session:
        while len(i)<max_len:
            i.insert(0,0)
    
    return np.asarray(list_session),np.asarray(list_label)

def get_dict_items(url_file):
    list_items = []
    dct_items = {0:0}
    df = pd.read_json(url_file)
    df = df.drop(df[df['videoids'].map(len) > 50].index)
    df = df.tail(10000)
    for i in df['videoids']:
        for j in i:
            list_items.append(j)
    for index,i in enumerate(set(list_items)):
        dct_items[i]=index+1
    return dct_items
