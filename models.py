"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


NUM_NER_TAGS = 52
NUM_POS_TAGS = 21


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
                 use_token=False, token_embed_size=0, use_exact=False):
        super(BiDAF, self).__init__()
        # Use character embeddings if fed into the BiDAF model
        self.use_char_embeddings = True if char_vectors is not None else False
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        # Variable to keep track of original hidden size
        final_hidden_size = hidden_size
        # If using character embeddings, feed through char-CNN to get word-level embeddings
        if self.use_char_embeddings:
            # Using char_out_size = hidden size as in original BiDAF paper
            self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                                 char_out_size=hidden_size,
                                                 drop_prob=drop_prob)
            # If concat [embed, char_embed], then final_hidden_size = hidden_size + char_out_size
            final_hidden_size += hidden_size  # since char_out_size = hidden_size

        # Now, account for tagged features
        final_context_hidden_size = final_hidden_size
        # If using NER and POS, feed through embedding
        self.use_token = use_token
        if self.use_token:
            # If x_emb -> [x_emb, x_pos, x_ner]; each word gets associated pos & ner
            if token_embed_size > 0:
                # Embed tokens:
                self.enc_ner = layers.TokenEncoder(num_tags=NUM_NER_TAGS, embed_size=token_embed_size,
                                                   drop_prob=drop_prob, use_embed=True)
                self.enc_pos = layers.TokenEncoder(num_tags=NUM_POS_TAGS, embed_size=token_embed_size,
                                                   drop_prob=drop_prob, use_embed=True)
                final_context_hidden_size += 2 * token_embed_size
            else:
                # One-hot-encode tokens:
                self.enc_ner = layers.TokenEncoder(num_tags=NUM_NER_TAGS)
                self.enc_pos = layers.TokenEncoder(num_tags=NUM_POS_TAGS)
                final_context_hidden_size += NUM_NER_TAGS + NUM_POS_TAGS
        # If using exact features
        self.use_exact = use_exact
        if self.use_exact:
            # 3 new features: exact_orig, exact_uncased, exact_lemma
            final_context_hidden_size += 3

        # Projection layer to decrease dimensions if extra features used
        self.project = nn.Linear(final_context_hidden_size, final_hidden_size, bias=False)

        # Highway Layer now outside of the Embedding layer...
        # - Allows concatenated word+char vector to be fed into Highway Layer if needed

        self.hwy = layers.HighwayEncoder(num_layers=2,
                                         hidden_size=final_hidden_size)

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

    def forward(self, cw_idxs, qw_idxs, cc_idxs=None, qc_idxs=None, ner_idxs=None, pos_idxs=None,
                exact_orig=None, exact_uncased=None, exact_lemma=None):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        if self.use_char_embeddings:
            cc_emb = self.char_emb(cc_idxs)  # (batch_size, c_len, hidden_size)
            qc_emb = self.char_emb(qc_idxs)  # (batch_size, q_len, hidden_size)
            c_emb = torch.cat([c_emb, cc_emb], dim=2)  # (batch_size, c_len, final_context_hidden_size = 2*hidden_size)
            q_emb = torch.cat([q_emb, qc_emb], dim=2)  # (batch_size, q_len, final_hidden_size = 2 * hidden_size)

        if self.use_token:
            ner_emb = self.enc_ner(ner_idxs)    # (batch_size, c_len, {token_embed_size OR NUM_NER_TAGS})
            pos_emb = self.enc_pos(pos_idxs)    # (batch_size, c_len, {token_embed_size OR NUM_POS_TAGS})
            print("c_emb has shape:", c_emb.shape)
            print("ner_emb has shape:", ner_emb.shape)
            print("pos_emb has shape:", pos_emb.shape)
            c_emb = torch.cat([c_emb, ner_emb, pos_emb], dim=2)
            # -> (batch_size, c_len, final_context_hidden_size += {2 * token_embed_size OR (NUM_NER_TAGS+NUM_POS_TAGS)})

        if self.use_exact:
            # exact_{orig, uncased, lemma} all have dimensions: (batch_size, c_len)
            exact_orig = torch.unsqueeze(exact_orig, dim=2).float()
            exact_uncased = torch.unsqueeze(exact_uncased, dim=2).float()
            exact_lemma = torch.unsqueeze(exact_lemma, dim=2).float()
            print("c_emb has shape:", c_emb.shape)
            print("exact_orig has shape:", exact_orig.shape)
            print("exact_uncased has shape:", exact_uncased.shape)
            print("exact_lemma has shape:", exact_lemma.shape)
            # -> (batch_size, c_len, 1)
            c_emb = torch.cat([c_emb, exact_orig, exact_uncased, exact_lemma], dim=2)
            # -> (batch_size, c_len, final_context_hidden_size += 3)

        print("c_emb shape after features added:", c_emb.shape)

        # Project context word embeddings from final_context_hidden_size -> final_hidden_size
        if self.use_exact or self.use_token:
            c_emb = self.project(c_emb)  # (batch_size, c_len, final_hidden_size)

        print("c_emb shape after projection layer:", c_emb.shape)

        c_emb = self.hwy(c_emb)  # (batch_size, c_len, final_hidden_size)
        q_emb = self.hwy(q_emb)  # (batch_size, q_len, final_hidden_size)

        print("c_emb shape after hwy layer:", c_emb.shape)
        print("q_emb shape:", q_emb.shape)

        # Adjust final_context_hidden_size -> final_hidden_size in enc layer
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * final_hidden_size)
        print("Passed q encoder!")
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * final_hidden_size)


        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * final_hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * final_hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

