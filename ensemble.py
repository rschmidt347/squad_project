"""
Generate ensemble submission by majority vote.
"""

import argparse
import glob
import pandas as pd


parser = argparse.ArgumentParser('Get args for ensemble script')

parser.add_argument('--split',
                    type=str,
                    default='dev',
                    choices=('dev', 'test'),
                    help='Split to use for ensembling.')
parser.add_argument('--sub_file',
                    type=str,
                    default='val_submission.csv',
                    help='Name for submission file.')
parser.add_argument('--file_to_omit',
                    type=str,
                    default='none',
                    help='Allow specification of file to omit')

args = parser.parse_args()

source_folder = './save/' + f'{args.split}' + '_submissions/'
stats_file = f'{args.split}' + '_stats.csv'

stats = pd.read_csv(stats_file)
stats_sub = stats[(stats['F1'] >= 65) & (stats['TestName'] != 'none') & (stats['TestName'] != args.file_to_omit)]
by_best_F1 = stats_sub.sort_values(by='F1', ascending=False)
file_best_F1 = source_folder + by_best_F1['TestName'].iloc[0] + '.csv'
filenames = list(stats_sub['TestName'])
filenames = [source_folder + file + '.csv' for file in filenames]

data = []
is_first_file = True
for filename in glob.glob(source_folder + '*.csv'):
    if filename in filenames:
        df = pd.read_csv(filename, keep_default_na=False)
        if is_first_file:
            df = df.rename(columns={'Predicted': filename})
            is_first_file = False
        else:
            df = df.rename(columns={'Predicted': filename})
            df = df[filename]
        data.append(df)

df_all = pd.concat(data, axis=1)
preds_best_F1 = df_all[file_best_F1]


def get_pred(row):
    pred = row.loc[file_best_F1]
    counts = row.value_counts(dropna=False)
    top_count = counts[0]
    if top_count == 1:
        return pred
    top_preds = list(counts[counts == top_count].index)
    if pred in top_preds:
        return pred
    return top_preds[0]


preds = df_all.apply(get_pred, axis=1)
d = {'Id': list(df_all['Id']), 'Predicted': preds.values}
output = pd.DataFrame(data=d)

out_file = source_folder + args.sub_file
output.to_csv(out_file, index=False)
