"""
Generate ensemble submission by majority vote.
"""

import argparse


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

args = parser.parse_args()

source_folder = './save/' + f'{args.split}' + '_submissions/'
stats_file = f'{args.split}' + '_stats.csv'





