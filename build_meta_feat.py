"""
Process file with additional features: lemma, NER, POS
"""
import ujson as json
import numpy as np
import spacy
from tqdm import tqdm
from collections import Counter
from setup import url_to_data_path, convert_idx, get_embedding, save, is_answerable
from args import get_add_feat_args


def load_pos_ner():
    ner_tags = ['PERSON', 'LOCATION', 'ORGANIZATION', 'MISC', 'MONEY', 'NUMBER', 'ORDINAL',
                'PERCENT', 'DATE', 'TIME', 'DURATION', 'SET', 'O']
    pos_tags = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN',
                'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '-LRB-', '-RRB-', '.', ',', '``',
                '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD',
                'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']

    # Build dictionaries mapping NER and POS to indices
    ner2idx_dict = dict(zip(ner_tags, range(2, len(ner_tags) + 2)))
    pos2idx_dict = dict(zip(pos_tags, range(2, len(pos_tags) + 2)))

    # Special cases for NULL and unseen POS/NER, just in case
    ner2idx_dict["--NULL--"] = 0
    ner2idx_dict["--OOV--"] = 1
    pos2idx_dict["--NULL--"] = 0
    pos2idx_dict["--OOV--"] = 1
    return ner2idx_dict, pos2idx_dict


def fix_quotes(token_list):
    token_list = ['" ' if token == "''" else token for token in token_list]
    token_list = ['" ' if token == "``" else token for token in token_list]
    return token_list


def process_file_w_add(filename, data_type, word_counter, char_counter):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = fix_quotes(para["context_tokens"])
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])

                lemma = fix_quotes(para["lemma"])
                ner = para["ner"]
                pos = para["pos"]

                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = fix_quotes(qa["ques_tokens"])
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    qlemma_tokens = fix_quotes(qa["qlemma"])
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens,
                               "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars,
                               "lemma_tokens": lemma,
                               "ner_tokens": ner,
                               "pos_tokens": pos,
                               "qlemma_tokens": qlemma_tokens,
                               "y1s": y1s,
                               "y2s": y2s,
                               "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "question": ques,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
        print(f"{len(examples)} questions in total")
    return examples, eval_examples


def build_features_added(args, examples, data_type, out_file, word2idx_dict, char2idx_dict,
                         ner2idx_dict, pos2idx_dict, is_test=False):
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    exact_orig_feat = []
    exact_uncased_feat = []
    exact_lemma_feat = []
    ner_idxs = []
    pos_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        def _get_ner(ner):
            if ner in ner2idx_dict:
                return ner2idx_dict[ner]
            return 1

        def _get_pos(pos):
            if pos in pos2idx_dict:
                return pos2idx_dict[pos]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        exact_orig = np.zeros([para_limit], dtype=np.int32)
        exact_uncased = np.zeros([para_limit], dtype=np.int32)
        exact_lemma = np.zeros([para_limit], dtype=np.int32)

        ner_idx = np.zeros([para_limit], dtype=np.int32)
        pos_idx = np.zeros([para_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
            # One hot encoding of exact match features
            exact_orig[i] = 1 if token in example["lemma_tokens"] else 0
            exact_uncased[i] = 1 if token.lower() in [ex.lower() for ex in example["ques_tokens"]] else 0
            exact_lemma[i] = 1 if example["lemma_tokens"][i] in example["qlemma_tokens"] else 0
        context_idxs.append(context_idx)
        exact_orig_feat.append(exact_orig)
        exact_uncased_feat.append(exact_uncased)
        exact_lemma_feat.append(exact_lemma)

        # Get indices for NER and POS tokens
        for i, token in enumerate(example["ner_tokens"]):
            ner_idx[i] = _get_ner(token)
        ner_idxs.append(ner_idx)

        for i, token in enumerate(example["pos_tokens"]):
            pos_idx[i] = _get_pos(token)
        pos_idxs.append(pos_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             exact_orig_feat=np.array(exact_orig_feat),
             exact_uncased_feat=np.array(exact_uncased_feat),
             exact_lemma_feat=np.array(exact_lemma_feat),
             ner_idxs=np.array(ner_idxs),
             pos_idxs=np.array(pos_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file_w_add(args.train_w_add_file, "train", word_counter, char_counter)

    # Load dictionaries mapping words and chars to indices
    with open(args.word2idx_file, "r") as word2idx_file:
        word2idx_dict = json.load(word2idx_file)
    with open(args.char2idx_file, "r") as char2idx_file:
        char2idx_dict = json.load(char2idx_file)

    # Load dictionaries mapping NER and POS to indices
    ner2idx_dict, pos2idx_dict = load_pos_ner()

    # Process dev and test sets
    dev_examples, dev_eval = process_file_w_add(args.dev_w_add_file, "dev", word_counter, char_counter)
    build_features_added(args, train_examples, "train", args.train_rec_add_file, word2idx_dict, char2idx_dict,
                         ner2idx_dict, pos2idx_dict)
    dev_meta = build_features_added(args, dev_examples, "dev", args.dev_rec_add_file, word2idx_dict, char2idx_dict,
                                    ner2idx_dict, pos2idx_dict)
    if args.include_test_examples:
        test_examples, test_eval = process_file_w_add(args.test_w_add_file, "test", word_counter, char_counter)
        save(args.test_eval_w_add_file, test_eval, message="test eval w add")
        test_meta = build_features_added(args, test_examples, "test", args.test_rec_add_file,
                                         word2idx_dict, char2idx_dict, ner2idx_dict, pos2idx_dict, is_test=True)
        save(args.test_meta_w_add_file, test_meta, message="test meta w add")

    save(args.train_eval_w_add_file, train_eval, message="train eval w add")
    save(args.dev_meta_w_add_file, dev_eval, message="dev eval w add")
    save(args.ner2idx_file, ner2idx_dict, message="NER dictionary")
    save(args.pos2idx_file, pos2idx_dict, message="POS dictionary")
    save(args.dev_meta_w_add_file, dev_meta, message="dev meta w add")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_add_feat_args()

    # Import spacy language model
    nlp = spacy.blank("en")

    # Preprocess dataset
    pre_process(args_)