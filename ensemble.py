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
parser.add_argument('--out_dir',
                    type=str,
                    default='',
                    help='Name for out directory')
parser.add_argument('--file_to_omit',
                    type=str,
                    default='none',
                    help='Allow specification of file to omit')
parser.add_argument('--metric_name',
                    type=str,
                    default='F1',
                    choices=('EM', 'F1'),
                    help='Name of metric to determine tie breaking')
parser.add_argument('--threshold',
                    type=float,
                    default=65.0,
                    help='Threshold for models to include in ensemble')
parser.add_argument('--models_to_include',
                    type=str,
                    default=None,
                    help='Optional file specifying exact models to include')

args = parser.parse_args()
print(f'{args.out_dir}')

source_folder = './save/' + f'{args.split}' + '_submissions/'
stats_file = 'sub_stats.csv'

stats = pd.read_csv(stats_file)

mods_to_include = []
if args.models_to_include is not None:
    filename = './save/' + f'{args.split}' + '_submissions/' + f'{args.models_to_include}'
    with open(filename, 'r') as fh:
        lines = fh.read().splitlines()
        for line in lines:
            mods_to_include.append(line)
    stats_sub = stats[stats['TestName'].isin(mods_to_include)]
else:
    stats_sub = stats[(stats[args.metric_name] >= args.threshold) & (stats['TestName'] != 'none') &
                      (stats['TestName'] != args.file_to_omit)]

by_best_metric = stats_sub.sort_values(by=args.metric_name, ascending=False)
file_best_metric = source_folder + by_best_metric['TestName'].iloc[0] + '.csv'
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


def get_pred(row):
    pred = row.loc[file_best_metric]
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

out_file = source_folder + f'{args.out_dir}' + '/' + args.sub_file
output.to_csv(out_file, index=False)

mod_file = source_folder + f'{args.out_dir}' + '/log.txt'
with open(mod_file, 'w') as file_handler:
    for item in filenames:
        file_handler.write("{}\n".format(item))
