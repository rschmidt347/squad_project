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
    """
    def __init__(self, word_vectors, hidden_size,
                 drop_prob=0., rnn_type='LSTM', num_mod_layers=2, char_vectors=None):
        super(BiDAF, self).__init__()
        # Use character embeddings if fed into the BiDAF model
        self.use_char_embeddings = True if char_vectors is not None else False
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        # If using character embeddings, feed through char-CNN to get word-level embeddings
        if self.use_char_embeddings:
            # Using char_out_size = hidden size as in original BiDAF paper
            self.char_emb = layers.CharEmbedding(char_vectors=char_vectors,
                                                 char_out_size=hidden_size,
                                                 drop_prob=drop_prob)
            # If concat [embed, char_embed], then final_hidden_size = hidden_size + char_out_size
            final_hidden_size = 2 * hidden_size  # since char_out_size = hidden_size
        else:
            # Otherwise, hidden size just reflects word hidden size = hidden_size
            final_hidden_size = hidden_size

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

    def forward(self, cw_idxs, qw_idxs, cc_idxs=None, qc_idxs=None):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        if self.use_char_embeddings:
            cc_emb = self.char_emb(cc_idxs)  # (batch_size, c_len, hidden_size)
            qc_emb = self.char_emb(qc_idxs)  # (batch_size, q_len, hidden_size)
            c_emb = torch.cat([c_emb, cc_emb], dim=2)  # (batch_size, c_len, final_hidden_size = 2 * hidden_size)
            q_emb = torch.cat([q_emb, qc_emb], dim=2)  # (batch_size, q_len, final_hidden_size)

        c_emb = self.hwy(c_emb)  # (batch_size, c_len, final_hidden_size)
        q_emb = self.hwy(q_emb)  # (batch_size, q_len, final_hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * final_hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * final_hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * final_hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * final_hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAFWF(BiDAF):
    """Baseline BiDAF model with additional input features

    Based on the paper:
    "Reading Wikipedia to Answer Open-Domain Questions"
    by Danqi Chen, Adam Fisch, Jason Weston & Antoine Bordes
    (https://arxiv.org/pdf/1704.00051.pdf)
    """
    def __init__(self, word_vectors, hidden_size, feature_dict,
                 drop_prob=0., rnn_type='LSTM', num_mod_layers=2):
        super(BiDAFWF, self).__init__(word_vectors, hidden_size, drop_prob=0., rnn_type='LSTM', num_mod_layers=2)
        self.feature_dict = feature_dict
