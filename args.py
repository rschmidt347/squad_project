"""Command-line arguments for setup.py, train.py, test.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import argparse


def get_setup_args(parser=None):

    """Get arguments needed in setup.py."""
    is_base_setup = False
    if parser is None:
        parser = argparse.ArgumentParser('Download and pre-process SQuAD')
        is_base_setup = True

    add_common_args(parser)

    parser.add_argument('--train_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/train-v2.0.json')
    parser.add_argument('--dev_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/dev-v2.0.json')
    parser.add_argument('--test_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/test-v2.0.json')
    parser.add_argument('--glove_url',
                        type=str,
                        default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--dev_meta_file',
                        type=str,
                        default='./data/dev_meta.json')
    parser.add_argument('--test_meta_file',
                        type=str,
                        default='./data/test_meta.json')
    parser.add_argument('--word2idx_file',
                        type=str,
                        default='./data/word2idx.json')
    parser.add_argument('--char2idx_file',
                        type=str,
                        default='./data/char2idx.json')
    parser.add_argument('--answer_file',
                        type=str,
                        default='./data/answer.json')
    parser.add_argument('--para_limit',
                        type=int,
                        default=400,
                        help='Max number of words in a paragraph')
    parser.add_argument('--ques_limit',
                        type=int,
                        default=50,
                        help='Max number of words to keep from a question')
    parser.add_argument('--test_para_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a paragraph at test time')
    parser.add_argument('--test_ques_limit',
                        type=int,
                        default=100,
                        help='Max number of words in a question at test time')
    parser.add_argument('--char_dim',
                        type=int,
                        default=64,
                        help='Size of char vectors (char-level embeddings)')
    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')
    parser.add_argument('--glove_num_vecs',
                        type=int,
                        default=2196017,
                        help='Number of GloVe vectors')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=30,
                        help='Max number of words in a training example answer')
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='Max number of chars to keep from a word')
    parser.add_argument('--include_test_examples',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Process examples from the test set')

    if is_base_setup:
        args = parser.parse_args()

        return args


def get_train_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train a model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)
    add_feature_filepath_args(parser)

    parser.add_argument('--eval_steps',
                        type=int,
                        default=50000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.5,
                        help='Learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('NLL', 'EM', 'F1'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    # -- New parser argument --
    # - Optimizer
    parser.add_argument('--model_optimizer',
                        type=str,
                        default='Adadelta',
                        choices=('Adadelta', 'Adamax'),
                        help='Choice of optimizer: supports Adadelta or Adamax.')

    args = parser.parse_args()

    # Error and case handling on metric
    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    # Check RNN type and context/question features
    train_test_error_checker(args)

    # Error handling for optimizer
    if args.model_optimizer not in ('Adadelta', 'Adamax'):
        raise ValueError(f'Unrecognized optimizer: "{args.model_optimizer}" - pick "Adadelta" or "Adamax"')

    return args


def get_test_args():
    """Get arguments needed in test.py."""
    parser = argparse.ArgumentParser('Test a trained model on SQuAD')

    add_common_args(parser)
    add_train_test_args(parser)
    add_feature_filepath_args(parser)

    # Change dev -> val for GradeScope
    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'),
                        help='Split to use for testing.')
    parser.add_argument('--sub_file',
                        type=str,
                        default='submission.csv',
                        help='Name for submission file.')

    # Require load_path for test.py
    args = parser.parse_args()
    if not args.load_path:
        raise argparse.ArgumentError('Missing required argument --load_path')

    # Check for errors in new hyperparameters
    train_test_error_checker(args)

    return args


def get_add_feat_args():
    """Get args used by setup_meta_feat.py"""
    parser = argparse.ArgumentParser('Pre-process additional features files')

    get_setup_args(parser)
    add_feature_filepath_args(parser)

    parser.add_argument('--ner2idx_file',
                        type=str,
                        default='./data/ner2idx.json')
    parser.add_argument('--pos2idx_file',
                        type=str,
                        default='./data/pos2idx.json')

    args = parser.parse_args()

    return args


def add_feature_filepath_args(parser):
    """List of filenames for datasets with added features"""
    # 1) Data files with tokens for context only
    # - spacy files
    parser.add_argument('--train_w_add_file',
                        type=str,
                        default='./data/train_w_spacy.json')
    parser.add_argument('--dev_w_add_file',
                        type=str,
                        default='./data/dev_w_spacy.json')
    parser.add_argument('--test_w_add_file',
                        type=str,
                        default='./data/test_w_spacy.json')
    # - .npz record files
    parser.add_argument('--train_w_add_record_file',
                        type=str,
                        default='./data/train_w_add_rec.npz')
    parser.add_argument('--dev_w_add_record_file',
                        type=str,
                        default='./data/dev_w_add_rec.npz')
    parser.add_argument('--test_w_add_record_file',
                        type=str,
                        default='./data/test_w_add_rec.npz')
    # - .json evaluation files
    parser.add_argument('--train_w_add_eval_file',
                        type=str,
                        default='./data/train_w_add_eval.json')
    parser.add_argument('--dev_w_add_eval_file',
                        type=str,
                        default='./data/dev_w_add_eval.json')
    parser.add_argument('--test_w_add_eval_file',
                        type=str,
                        default='./data/test_w_add_eval.json')
    # - Meta files for construction
    parser.add_argument('--dev_w_add_meta_file',
                        type=str,
                        default='./data/dev_w_add_meta.json')
    parser.add_argument('--test_w_add_meta_file',
                        type=str,
                        default='./data/test_w_add_meta.json')

    # 2) Data files with tokens for context and questions
    # - .npz record files
    parser.add_argument('--train_qtok_record_file',
                        type=str,
                        default='./data/train_qtok_rec.npz')
    parser.add_argument('--dev_qtok_record_file',
                        type=str,
                        default='./data/dev_qtok_rec.npz')
    parser.add_argument('--test_qtok_record_file',
                        type=str,
                        default='./data/test_qtok_rec.npz')
    # - .json evaluation files
    parser.add_argument('--train_qtok_eval_file',
                        type=str,
                        default='./data/train_qtok_eval.json')
    parser.add_argument('--dev_qtok_eval_file',
                        type=str,
                        default='./data/dev_qtok_eval.json')
    parser.add_argument('--test_qtok_eval_file',
                        type=str,
                        default='./data/test_qtok_eval.json')


def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')
    parser.add_argument('--word_emb_file',
                        type=str,
                        default='./data/word_emb.json')
    parser.add_argument('--char_emb_file',
                        type=str,
                        default='./data/char_emb.json')
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./data/dev_eval.json')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='./data/test_eval.json')
    # New argument: whether or not to use default files
    parser.add_argument('--use_default_task_files',
                        type=bool,
                        default=True,
                        help="Flag to automatically switch over to the \
                        correct files based on input.")


def add_train_test_args(parser):
    """Add arguments common to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--max_ans_len',
                        type=int,
                        default=15,
                        help='Maximum length of a predicted answer.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')
    parser.add_argument('--use_squad_v2',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use SQuAD 2.0 (unanswerable) questions.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    # -- New parser arguments --
    # 1) Hyperparameters
    # - RNN type: (LSTM, GRU)
    parser.add_argument('--rnn_type',
                        type=str,
                        default='LSTM',
                        choices=('LSTM', 'GRU'),
                        help='RNN encoder type for the RNNEncoder layer.')
    # - Number of encoder model hidden layers
    parser.add_argument('--num_mod_layers',
                        type=int,
                        default=2,
                        help='Number of RNN layers in encoder "mod" modeling layer.')
    # 2) Character embeddings
    # - Flag to use character embeddings
    parser.add_argument('--use_char_embeddings',
                        type=bool,
                        default=False,
                        help='Flag to use character embeddings in the BiDAF model.')
    # 3) Token features
    # - Options for use of exact match features
    parser.add_argument('--use_exact',
                        type=str,
                        default=False,
                        help='Whether to add exact match features. Can specify context only or context & question.')
    # - Options for use of token features (POS, NER)
    parser.add_argument('--use_token',
                        type=str,
                        default=False,
                        help='Whether to add token features (POS, NER). Can specify context only or context & question.')
    # - Flag for size of embedding for NER and POS
    parser.add_argument('--token_embed_size',
                        type=int,
                        default=0,
                        help='Size of embedding for NER and POS.')
    # - Flag for use of projection of embedding with added features
    parser.add_argument('--use_projection',
                        type=lambda s: s.lower() in ('yes', 'y', 'true', 't', '1'),
                        default=False,
                        help='Whether to use projection when adding features')
    # - Flag to one-hot-encode tokens
    parser.add_argument('--token_one_hot',
                        type=bool,
                        default=False,
                        help='Whether to append raw token index or convert to one-hot if using tokens.')


def train_test_error_checker(args):
    """Check for input errors for args involving new model options."""
    # Error handling for RNN type
    if args.rnn_type not in ('LSTM', 'GRU'):
        raise ValueError(f'Unrecognized RNN type: "{args.rnn_type}" - pick "LSTM" or "GRU"')

    # Error handling for added features
    if args.use_token not in (False, 'c', 'cq'):
        raise ValueError(f'Unrecognized option for token use: "{args.use_token}" - pick "False", "c", or "cq"')
    if args.use_exact not in (False, 'c', 'cq'):
        raise ValueError(f'Unrecognized option for EM use: "{args.use_exact}" - pick "False", "c", or "cq"')
