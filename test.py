"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, collate_fn_cq, SQuAD


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    # Option to use character embeddings
    if args.use_char_embeddings:
        char_vectors = util.torch_from_json(args.char_emb_file)

    # Get model
    log.info('Building model...')
    # Take note of extra features
    token_flag = True if args.use_token in ('c', 'cq') else False
    exact_flag = True if args.use_exact in ('c', 'cq') else False
    context_and_question_flag = True if args.use_token == 'cq' else False
    # Switch over to proper data files if none specified
    if args.use_default_task_files:
        args, log = switch_to_default_files(args, log)

    model = BiDAF(word_vectors=word_vectors,
                  char_vectors=char_vectors if args.use_char_embeddings else None,
                  hidden_size=args.hidden_size,
                  rnn_type=args.rnn_type,
                  num_mod_layers=args.num_mod_layers,
                  use_token=token_flag,
                  use_exact=exact_flag,
                  context_and_question=context_and_question_flag,
                  token_embed_size=args.token_embed_size,
                  use_projection=args.use_projection)

    model = nn.DataParallel(model, gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = util.load_model(model, args.load_path, gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2,
                    use_token=token_flag,
                    use_exact=exact_flag,
                    context_and_question=context_and_question_flag)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_cq if context_and_question_flag else collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for example in data_loader:

            cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids = example[:7]

            ner_idxs, pos_idxs, qner_idxs, qpos_idxs = None, None, None, None
            exact_orig, exact_uncased, exact_lemma = None, None, None
            qexact_orig, qexact_uncased, qexact_lemma = None, None, None
            if context_and_question_flag:
                if token_flag:
                    ner_idxs, pos_idxs, qner_idxs, qpos_idxs = example[7:11]
                    ner_idxs = ner_idxs.to(device)
                    pos_idxs = pos_idxs.to(device)
                    qner_idxs = qner_idxs.to(device)
                    qpos_idxs = qpos_idxs.to(device)
                    if exact_flag:
                        # Token features present, so splice example at later index
                        exact_orig, exact_uncased, exact_lemma, qexact_orig, qexact_uncased, qexact_lemma = example[
                                                                                                            11:]
                        exact_orig = exact_orig.to(device)
                        exact_uncased = exact_uncased.to(device)
                        exact_lemma = exact_lemma.to(device)
                        qexact_orig = qexact_orig.to(device)
                        qexact_uncased = qexact_uncased.to(device)
                        qexact_lemma = qexact_lemma.to(device)
                else:
                    if exact_flag:
                        # Token features not present, so splice example at earlier index
                        exact_orig, exact_uncased, exact_lemma, qexact_orig, qexact_uncased, qexact_lemma = example[
                                                                                                            7:]
                        exact_orig = exact_orig.to(device)
                        exact_uncased = exact_uncased.to(device)
                        exact_lemma = exact_lemma.to(device)
                        qexact_orig = qexact_orig.to(device)
                        qexact_uncased = qexact_uncased.to(device)
                        qexact_lemma = qexact_lemma.to(device)
            else:
                if token_flag:
                    ner_idxs, pos_idxs = example[7:9]
                    ner_idxs = ner_idxs.to(device)
                    pos_idxs = pos_idxs.to(device)
                    if exact_flag:
                        # Token features present, so splice example at later index
                        exact_orig, exact_uncased, exact_lemma = example[9:]
                        exact_orig = exact_orig.to(device)
                        exact_uncased = exact_uncased.to(device)
                        exact_lemma = exact_lemma.to(device)
                else:
                    if exact_flag:
                        # Token features present, so splice example at earlier index
                        exact_orig, exact_uncased, exact_lemma = example[7:]
                        exact_orig = exact_orig.to(device)
                        exact_uncased = exact_uncased.to(device)
                        exact_lemma = exact_lemma.to(device)

            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            if args.use_char_embeddings:
                cc_idxs = cc_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            if args.use_char_embeddings:
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs,
                                       ner_idxs=ner_idxs, pos_idxs=pos_idxs,
                                       exact_orig=exact_orig, exact_uncased=exact_uncased, exact_lemma=exact_lemma,
                                       qner_idxs=qner_idxs, qpos_idxs=qpos_idxs,
                                       qexact_orig=qexact_orig, qexact_uncased=qexact_uncased, qexact_lemma=qexact_lemma)
            else:
                log_p1, log_p2 = model(cw_idxs, qw_idxs,
                                       ner_idxs=ner_idxs, pos_idxs=pos_idxs,
                                       exact_orig=exact_orig, exact_uncased=exact_uncased, exact_lemma=exact_lemma,
                                       qner_idxs=qner_idxs, qpos_idxs=qpos_idxs,
                                       qexact_orig=qexact_orig, qexact_uncased=qexact_uncased, qexact_lemma=qexact_lemma)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


def switch_to_default_files(args, log):
    """Update args and log to switch to default files if none specified"""
    if args.use_token == "c" or args.use_exact == "c":
        # Use added feature files for context only
        log.info('Using context-only feature files based on provided input...')
        log.info('To manually specify files, set --use_default_task_files to False.')
        for data_split in ['train', 'dev', 'test']:
            # .npz record files
            vars(args)[f'{data_split}_record_file'] = vars(args)[f'{data_split}' + '_w_add_record_file']
            # .json eval files
            vars(args)[f'{data_split}_eval_file'] = vars(args)[f'{data_split}' + '_w_add_eval_file']
        if not args.use_projection and args.token_embed_size == 0:
            log.info('Turning on projection to ensure dimension agreement...')
            vars(args)['use_projection'] = True
    elif args.use_token == "cq" or args.use_exact == "cq":
        # Use added feature files for context and question
        log.info('Using context & question feature files based on provided input...')
        log.info('To manually specify files, set --use_default_task_files to False.')
        for data_split in ['train', 'dev', 'test']:
            # .npz record files
            vars(args)[f'{data_split}_record_file'] = vars(args)[f'{data_split}' + '_qtok_record_file']
            # .json eval files
            vars(args)[f'{data_split}_eval_file'] = vars(args)[f'{data_split}' + '_qtok_eval_file']

    return args, log


if __name__ == '__main__':
    main(get_test_args())
