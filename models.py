"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        rnn_type (str): RNN architecture used for encoder layer; one of 'LSTM' or 'GRU'.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        use_token (bool): Flag for using token features (NER, POS)
        token_embed_size (int): Size of embedding for NER/POS; 0 if using one-hot-encoding
        use_exact (bool): Flag for using exact match features (original, uncased, lemma)
    """
    def __init__(self, word_vectors, hidden_size,
                 drop_prob=0., rnn_type='LSTM', num_mod_layers=2, char_vectors=None,
                 use_token=False, use_exact=False, context_and_question=False,
                 token_embed_size=0, use_projection=False, token_one_hot=False,
                 num_ner_tags=52, num_pos_tags=21):
        super(BiDAF, self).__init__()
        # 0) Use character embeddings if fed into the BiDAF model
        self.use_char_embeddings = True if char_vectors is not None else False

        # 1) Word embedding layer
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        # 2) If using character embeddings, feed through char-CNN to get word-level embeddings
        # - Keep track of new hidden size if adding character vectors
        final_hidden_size = hidden_size
        if self.use_char_embeddings:
            # Using char_out_size = hidden size as in original BiDAF paper
            self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                                 char_out_size=hidden_size,
                                                 drop_prob=drop_prob)
            # If concat [embed, char_embed], then final_hidden_size = hidden_size + char_out_size
            final_hidden_size += hidden_size  # since char_out_size = hidden_size

        # 3) Now, account for tagged features if needed
        # - Keep track of new hidden size if adding tagged features
        final_doc_hidden_size = final_hidden_size
        self.use_token = use_token
        self.token_one_hot = token_one_hot
        # 3 a) Token features: POS, NER
        if self.use_token:
            if token_embed_size > 0:
                # If embedding size specified, embed the tokens
                self.enc_ner = layers.TokenEncoder(num_tags=num_ner_tags,
                                                   embed_size=token_embed_size,
                                                   drop_prob=drop_prob)
                self.enc_pos = layers.TokenEncoder(num_tags=num_pos_tags,
                                                   embed_size=token_embed_size,
                                                   drop_prob=drop_prob)
                final_doc_hidden_size += 2 * token_embed_size
            elif token_one_hot:
                # If one hot flag, convert raw token indices to one-hot & append
                self.enc_ner = layers.TokenEncoder(num_tags=num_ner_tags, token_one_hot=True)
                self.enc_pos = layers.TokenEncoder(num_tags=num_pos_tags, token_one_hot=True)
                final_doc_hidden_size += num_ner_tags + num_ner_tags
            else:
                # No embedding, simply append the index for ner, pos, (resp. qner, qpos)
                final_doc_hidden_size += 2
        # 3 b) Exact match features: original match, uncased match, lemma match
        self.use_exact = use_exact
        if self.use_exact:
            # Concatenate 3 binary features to each vector
            final_doc_hidden_size += 3
        # 3 c) Flag for whether features are in context and/or question
        self.context_and_question = context_and_question

        # 4) Projection layer to decrease dimensions if extra features used
        self.use_projection = use_projection
        if self.use_projection:
            self.project = nn.Linear(final_doc_hidden_size, final_hidden_size, bias=False)
        else:
            final_hidden_size = final_doc_hidden_size

        # 5) Highway Layer now outside of the Embedding layer...
        # - Allows concatenated word+char vector to be fed into Highway Layer if needed
        self.hwy = layers.HighwayEncoder(num_layers=2,
                                         hidden_size=final_hidden_size)

        # 6) Proceed with remainder of BiDAF model, accounting for new hidden sizes
        self.enc = layers.RNNEncoder(input_size=final_hidden_size,
                                     hidden_size=final_hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob,
                                     rnn_type=rnn_type)

        self.att = layers.BiDAFAttention(hidden_size=2 * final_hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * final_hidden_size,
                                     hidden_size=final_hidden_size,
                                     num_layers=num_mod_layers,
                                     drop_prob=drop_prob,
                                     rnn_type=rnn_type)

        self.out = layers.BiDAFOutput(hidden_size=final_hidden_size,
                                      drop_prob=drop_prob,
                                      rnn_type=rnn_type)

    def forward(self, cw_idxs, qw_idxs, cc_idxs=None, qc_idxs=None,
                ner_idxs=None, pos_idxs=None, exact_orig=None, exact_uncased=None, exact_lemma=None,
                qner_idxs=None, qpos_idxs=None, qexact_orig=None, qexact_uncased=None, qexact_lemma=None):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        if self.use_char_embeddings:
            cc_emb = self.char_emb(cc_idxs)  # (batch_size, c_len, hidden_size)
            c_emb = torch.cat([c_emb, cc_emb], dim=2)  # (batch_size, c_len, final_doc_hidden_size = 2*hidden_size)
            qc_emb = self.char_emb(qc_idxs)  # (batch_size, q_len, hidden_size)
            q_emb = torch.cat([q_emb, qc_emb], dim=2)  # (batch_size, q_len, final_doc_hidden_size = 2*hidden_size)

        if self.use_token:
            # NER, POS indices: (batch_size, c_len)
            if self.token_one_hot:
                ner_idxs = self.enc_ner(ner_idxs).float()  # (batch_size, c_len, num_ner_tags)
                pos_idxs = self.enc_pos(pos_idxs).float()  # (batch_size, c_len, num_pos_tags)
            else:
                ner_idxs = torch.unsqueeze(ner_idxs, dim=2).float()  # -> (batch_size, c_len, 1)
                pos_idxs = torch.unsqueeze(pos_idxs, dim=2).float()  # -> (batch_size, c_len, 1)
            c_emb = torch.cat([c_emb, ner_idxs, pos_idxs], dim=2)
            # -> final output: (batch_size, c_len, final_doc_hidden_size += <token dims>)
            if self.context_and_question:
                if self.token_one_hot:
                    qner_idxs = self.enc_ner(qner_idxs).float()  # (batch_size, q_len, num_ner_tags)
                    qpos_idxs = self.enc_pos(qpos_idxs).float()  # (batch_size, q_len, num_pos_tags)
                else:
                    qner_idxs = torch.unsqueeze(qner_idxs, dim=2).float()  # -> (batch_size, q_len, 1)
                    qpos_idxs = torch.unsqueeze(qpos_idxs, dim=2).float()  # -> (batch_size, q_len, 1)
                q_emb = torch.cat([q_emb, qner_idxs, qpos_idxs], dim=2)
                # -> final output: (batch_size, q_len, final_doc_hidden_size += {2, 2 * token_embed_size})

        if self.use_exact:
            # exact_{orig, uncased, lemma} all have dimensions: (batch_size, c_len)
            exact_orig = torch.unsqueeze(exact_orig, dim=2).float()  # -> (batch_size, c_len, 1)
            exact_uncased = torch.unsqueeze(exact_uncased, dim=2).float()  # -> (batch_size, c_len, 1)
            exact_lemma = torch.unsqueeze(exact_lemma, dim=2).float()  # -> (batch_size, c_len, 1)
            c_emb = torch.cat([c_emb, exact_orig, exact_uncased, exact_lemma], dim=2)
            # -> final output: (batch_size, c_len, final_doc_hidden_size += 3)
            if self.context_and_question:
                qexact_orig = torch.unsqueeze(qexact_orig, dim=2).float()  # -> (batch_size, q_len, 1)
                qexact_uncased = torch.unsqueeze(qexact_uncased, dim=2).float()  # -> (batch_size, q_len, 1)
                qexact_lemma = torch.unsqueeze(qexact_lemma, dim=2).float()  # -> (batch_size, q_len, 1)
                q_emb = torch.cat([q_emb, qexact_orig, qexact_uncased, qexact_lemma], dim=2)
                # -> final_output: (batch_size, q_len, final_doc_hidden_size += 3)

        # Project context/question embeddings from final_doc_hidden_size -> final_hidden_size
        if self.use_projection:
            if self.use_exact or self.use_token:
                c_emb = self.project(c_emb)  # (batch_size, c_len, final_hidden_size)
                if self.context_and_question:
                    q_emb = self.project(q_emb)  # (batch_size, q_len, final_hidden_size)
        # else: let final_hidden_size = final_doc_hidden_size

        c_emb = self.hwy(c_emb)  # (batch_size, c_len, final_hidden_size)
        q_emb = self.hwy(q_emb)  # (batch_size, q_len, final_hidden_size)

        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * final_hidden_size)
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * final_hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * final_hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * final_hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

