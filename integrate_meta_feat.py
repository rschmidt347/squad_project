"""
Add input features from spaCy tokenizer to output of setup.py
by combining pre-processed data .txt file from DrQA script with
original json SQuAD data.
"""
import json
import pandas as pd
import argparse
from io import StringIO


def get_feat_args():
    parser = argparse.ArgumentParser('Integrate additional features')

    parser.add_argument('--orig_file',
                        type=str,
                        default='./data/dev-v2.0.json')

    parser.add_argument('--proc_file',
                        type=str,
                        default='./data/dev-v2.0-qtok-spacy.txt')

    parser.add_argument('--out_file',
                        type=str,
                        default='./data/dev_qtok_spacy.json')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args_ = get_feat_args()

    with open(args_.orig_file, 'r') as dev_orig:
        exs_orig = dev_orig.read()
        exs_orig_json = json.loads(exs_orig)
        dat_orig = exs_orig_json['data']

    print("Finished reading in original file")

    with open(args_.proc_file, 'r') as dev_proc:
        exs_proc = dev_proc.readlines()
        exs_proc = [ex.rstrip() for ex in exs_proc]
        dfs_proc_list = [pd.read_json(StringIO(ex), orient='index').T for ex in exs_proc]
        dfs_proc_long = pd.concat(dfs_proc_list)
        dfs_proc_long.reset_index(drop=True, inplace=True)

    print("Finished reading in processed file")

    # Add list of question ids for every paragraph
    for article in dat_orig:
        for para in article['paragraphs']:
            qas_list = para['qas']
            q_ids_list = [q['id'] for q in qas_list]
            para['q_ids'] = q_ids_list

    print("Finished adding list of question ids for every paragraph")


    def add_feat(row):
        cur_id = row['id']
        for article in dat_orig:
            for para in article['paragraphs']:
                if cur_id in para['q_ids']:
                    if 'lemma' not in para:
                        para['lemma'] = row['lemma']
                        para['pos'] = row['pos']
                        para['ner'] = row['ner']
                        para['context_tokens'] = row['document']
                    for qa in para['qas']:
                        if cur_id == qa['id']:
                            qa['ques_tokens'] = row['question']
                            qa['qlemma'] = row['qlemma']
                            qa['qner'] = row['qner']
                            qa['qpos'] = row['qpos']
                            break


    dfs_proc_long.apply(add_feat, axis=1)

    print("Finished adding features to original data")

    with open(args_.out_file, 'w') as outfile:
        json.dump(exs_orig_json, outfile)

    print("Outputted new json file with features")