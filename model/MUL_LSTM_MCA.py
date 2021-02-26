# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF
import numpy as np
from model.mca import MCA_ED

class MUL_LSTM_MCA(nn.Module):
    """
    char lstm + word lstm + crf
    """
    def __init__(self, data):
        super(MUL_LSTM_MCA, self).__init__()
        print('build MUL_LSTM_MCA network...')
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.bilstm_flag = data.HP_bilstm

        # word represent = char lstm represent + word emb
        self.char_embeddings = nn.Embedding(data.char_alphabet.size(), data.char_emb_dim)
        if data.pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.char_alphabet.size(), data.char_emb_dim)))
        self.char_drop = nn.Dropout(data.HP_dropout)  # emb -> dropout -> char lstm
        self.char_lstm = nn.LSTM(data.char_emb_dim, data.HP_char_hidden_dim // 2, num_layers=1, batch_first=True,
                                 bidirectional=True)

        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), data.word_emb_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(), data.word_emb_dim)))
        self.word_drop = nn.Dropout(data.HP_dropout)  # [char_presentation, word embedding] -> dropout

        # word seq lstm
        self.input_size = data.word_emb_dim + data.HP_char_hidden_dim
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.word_lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True,
                                 bidirectional=data.HP_bilstm)
        self.droplstm = nn.Dropout(data.HP_dropout)  # word seq lstm out -> dropout

        # object
        # self.object_embeddings = nn.Embedding(data.object_alphabet.size(), data.object_emb_dim, padding_idx=0)
        if data.pretrain_object_embedding is not None:
            self.object_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(data.pretrain_object_embedding),
                                                                  freeze=True)

        # self.object_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_object_embedding))
        else:
            self.object_embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(self.random_embedding(data.object_alphabet.size(), data.object_emb_dim)), freeze=True)

        # topic_embed_dim to the dim as text hidden dim
        self.obj_feat_linear = nn.Linear(data.object_emb_dim, data.HP_hidden_dim)

        # MCA_ED
        self.mca_params = {
            'HIDDEN_SIZE': data.HP_hidden_dim,
            'DROPOUT_R': data.HP_MCA_dropout,
            'MULTI_HEAD': data.HP_multi_head,
            'FF_SIZE': data.HP_hidden_dim * 4,
            'HIDDEN_SIZE_HEAD': int(data.HP_hidden_dim / data.HP_multi_head),
            'LAYER': data.HP_SGA_layer,
        }
        self.mca = MCA_ED(self.mca_params)

        # hidden to tag, add the start and end tag
        self.cat_bn = nn.BatchNorm1d(self.mca_params['HIDDEN_SIZE'])
        self.hidden2tag = nn.Linear(self.mca_params['HIDDEN_SIZE'], data.label_alphabet_size + 2)

        # crf
        self.crf = CRF(data.label_alphabet_size, self.gpu)


    def _get_lstm_features(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                           char_seq_recover, object_inputs, object_mask):
        """
        char-> char embedding -> dropout -> char lstm -> char representation
            word -> word embedding
            [char representation, word embedding] -> dropout -> word lstm -> dropout -> Linear -> tagscores
        :param word_inputs: (batch_size, sent_len)
        :param feature_inputs:  [(batch_size, sent_len), ...] list of variables
        :param word_seq_lengths: list of batch_size, (batch_size, 1)
        :param char_inputs: (batch_size * sent_len, word_length)
        :param char_seq_lengths: (batch_size * sent_len, 1)
        :param char_seq_recover: variable which records the char order information, used to recover char order
        :return:
            variable(batch_size,sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        char_batch_size = char_inputs.size(0)
        max_object_nb = object_inputs.size(1)

        text_mask = (word_inputs == 0).unsqueeze(1).unsqueeze(2)
        object_mask = (object_inputs == 0).unsqueeze(1).unsqueeze(2)

        # char -> emb -> drop
        char_embeds = self.char_drop(
            self.char_embeddings(char_inputs))  # (batch_size * sent_len, word_length, char_emb_dim)
        char_hidden = None
        # -> char lstm,
        pack_char_input = pack_padded_sequence(char_embeds, char_seq_lengths.cpu().numpy(), batch_first=True)
        char_rnn_out, char_hidden = self.char_lstm(pack_char_input, char_hidden)
        # last hiddens
        ## char_hidden = (h_t, c_t)
        #  char_hidden[0] = h_t = (2, batch_size, lstm_dimension)
        char_features = char_hidden[0].transpose(1, 0).contiguous().view(char_batch_size,
                                                                         -1)  # (batch_size * sent_len, char_hidden_dim)
        char_features = char_features[char_seq_recover]
        # cat char_hidden_dim for every char in a word
        char_features = char_features.view(batch_size, sent_len, -1)  # (batch_size, sent_len, char_hidden_dim)

        # word -> word emb
        word_embs = self.word_embeddings(word_inputs)
        # concat-> word represent
        word_represent = torch.cat([word_embs, char_features], 2)
        word_represent = self.word_drop(word_represent)  # (batch_size, sent_len, char_hidden_dim + word_emb_dim)

        # -> word seq lstm
        packed_word = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.word_lstm(packed_word, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        text_feat = self.droplstm(lstm_out)  # (batch_size, sent_len, hidden_dim) X

        # object emb dim to object hidden dim
        object_embs = self.object_embeddings(object_inputs)  # (batch_size, max_obj_nb, obj_emb_dim) Y
        object_feat = self.obj_feat_linear(object_embs)  # (batch_size, max_obj_nb, obj_hidden_dim)

        # MAC
        # text_feats: (b, seq_t, dim)
        # object_feats: (b, seq_o, dim)
        _, object_feat = self.mca(text_feat, object_feat, text_mask, object_mask)

        # concat
        final_feature = text_feat + object_feat
        final_feature = final_feature.transpose(2, 1).contiguous()  # (batch_size, hidden_dim + hidden_dim, sent_len)
        final_feature = self.cat_bn(final_feature)
        final_feature = final_feature.transpose(2, 1).contiguous()  # (batch_size, sent_len, hidden_dim + hidden_dim)
        # -> tagscore
        outputs = self.hidden2tag(final_feature)
        return outputs

    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                       char_seq_recover, batch_label, mask, object_inputs, object_mask):
        """
            char-> char embedding -> dropout -> char lstm -> char representation
            word -> word embedding
            [char representation, word embedding] -> dropout -> word lstm -> dropout -> Linear -> tagscores
            - crf -> loss
        :param word_inputs: (batch_size, sent_len)
        :param feature_inputs:  [(batch_size, sent_len), ...] list of variables
        :param word_seq_lengths: list of batch_size, (batch_size, 1)
        :param char_inputs: (batch_size * sent_len, word_length)
        :param char_seq_lengths: (batch_size * sent_len, 1)
        :param char_seq_recover: variable which records the char order information, used to recover char order
        :param batch_label: (batch_size, sent_len)
        :param mask: (batch_size, sent_len)
        :return:
            variable(batch_size,sent_len, hidden_dim)
        """

        outs = self._get_lstm_features(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                       char_seq_recover, object_inputs, object_mask)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        # crf
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)  # batch,
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                mask, object_inputs, object_mask):
        outs = self._get_lstm_features(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                       char_seq_recover, object_inputs, object_mask)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return tag_seq

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover, mask, nbest):
        outs = self._get_lstm_features(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                       char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq


    # Masking
    def make_mask(self, feature):
        '''

        :param feature: feature(b, seq, dim)->(b, seq) -> (b, 1, seq, 1)
        :return:
        '''
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)