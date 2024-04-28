import pandas as pd
import numpy as np
from math import ceil
from collections import Counter
from tqdm.notebook import tqdm


def sort_df(df):
    df.sort_values(by='query_id', inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def drop_features(df):
    features2drop = []
    for el in df.columns:
        if df[el].unique().shape[0] == 1:
            features2drop.append(el)
            print(f"name: {el} | item: {df[el].unique().item()} | dtype: {df[el].dtype}")
    return df.drop(features2drop, axis=1)

def train_test(df, test_ratio=0.25):

    df.sort_values(by='query_id', inplace=True)
    q = df['query_id']
    count = Counter(q)
    train_sizes = {key: ceil((1 - test_ratio) * count[key]) for key in count}
    dataset_mask = np.zeros(df.shape[0], dtype=bool)
    i = 0
    for el in count:
        true_size = train_sizes[el]
        false_size = count[el] - true_size
        query_mask = np.random.permutation([True] * true_size + [False] * false_size)
        dataset_mask[i:i + count[el]] = query_mask
        i += count[el]
    df_train = df[dataset_mask].copy()
    df_test = df[~dataset_mask].copy()
    df_train.sort_values(by='query_id', inplace=True)
    df_train.reset_index(inplace=True, drop=True)
    df_test.sort_values(by='query_id', inplace=True)
    df_test.reset_index(inplace=True, drop=True)
    
    return df_train, df_test

def fix_class_dist(df):
    counter = Counter(df['query_id'].values)
    for q in tqdm(counter):
        df_q = df[df['query_id'] == q]
        ranks_in_q = Counter(df_q['rank'])
        if len(ranks_in_q) == 1:
            continue
        size = len(df_q['rank'])
        probability = {r: ranks_in_q[r] / size for r in ranks_in_q}
        max_value_key = max(probability, key=probability.get)
        max_values_per_r = int(probability[max_value_key] * size)
        for r in probability:
            if r == max_value_key:
                continue
            df_q_r = df_q[df_q['rank'] == r]
            samples_to_add = max_values_per_r - int(probability[r] * size)
            df = pd.concat([df, df_q_r.sample(n=samples_to_add, replace=True)], ignore_index=True)
    return df

def group_by(df):
    queries = df['query_id'].unique()
    qid_doc_map = {q: [] for q in queries}
    for q in queries:
        qid_doc_map[q] = df.index[df['query_id'] == q].tolist()
    return qid_doc_map

