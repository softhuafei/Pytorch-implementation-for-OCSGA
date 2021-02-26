# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np
from tqdm import tqdm

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized, max_sent_length,
                  sentence_classification=False, split_token='\t', char_padding_size=-1, char_padding_symbol='</pad>', object_file=None, object_alphabet=None):
    feature_num = len(feature_alphabets)
    in_lines = open(input_file,'r', encoding="utf8").readlines()
    img2object = {}
    if object_file is not None:
        # 读取所有的topics， img2topic = {'imgID', [object1, object2,...]}
        with open(object_file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().split(" ")
                imgID = line[0].split(".")[0]  # ID
                object = line[1:]
                img2object[imgID] = object

    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    objects = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []
    object_Ids = []

    ## if sentence classification data format, splited by \t
    if sentence_classification:
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split(split_token)
                sent = pairs[0]
                if sys.version_info[0] < 3:
                    sent = sent.decode('utf-8')
                original_words = sent.split()
                for word in original_words:
                    words.append(word)
                    if number_normalized:
                        word = normalize_word(word)
                    word_Ids.append(word_alphabet.get_index(word))
                    ## get char
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                        assert(len(char_list) == char_padding_size)
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)

                label = pairs[-1]
                label_Id = label_alphabet.get_index(label)
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx+1].split(']',1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                ## combine together and return, notice the feature/label as different format with sequence labeling task
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                    instence_texts.append([words, feat_list, chars, label])
                    instence_Ids.append([word_Ids, feat_Id, char_Ids,label_Id])
                words = []
                features = []
                chars = []
                char_Ids = []
                word_Ids = []
                feature_Ids = []
                label_Ids = []
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
            instence_texts.append([words, feat_list, chars, label])
            instence_Ids.append([word_Ids, feat_Id, char_Ids,label_Id])
            words = []
            features = []
            chars = []
            char_Ids = []
            word_Ids = []
            feature_Ids = []
            label_Ids = []

    else:
    ### for sequence labeling data format i.e. CoNLL 2003
        for line in in_lines:
            if len(line) > 2:  # 判断是否空行以及新的instance
                if len(line.strip().split('\t')) >= 2:  # 判断是否是IMAGE ID
                    pairs = line.strip().split()
                    word = pairs[0]
                    if sys.version_info[0] < 3:
                        word = word.decode('utf-8')
                    words.append(word)
                    if number_normalized:
                        word = normalize_word(word)
                    label = pairs[-1]
                    labels.append(label)
                    word_Ids.append(word_alphabet.get_index(word))
                    label_Ids.append(label_alphabet.get_index(label))
                    ## get features
                    feat_list = []
                    feat_Id = []
                    for idx in range(feature_num):
                        feat_idx = pairs[idx+1].split(']',1)[-1]
                        feat_list.append(feat_idx)
                        feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                    features.append(feat_list)
                    feature_Ids.append(feat_Id)
                    ## get char
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                        assert(len(char_list) == char_padding_size)
                    else:
                        ### not padding
                        pass
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)
                else:
                    # IMAGE ID
                    if line.strip().split(':')[0] == 'IMGID':
                        objects = img2object.get(line.strip().split(':')[1], [])
                        for object in objects:
                            object_Ids.append(object_alphabet.get_index(object))
            else:
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                    instence_texts.append([words, features, chars, labels, objects])
                    instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids, object_Ids])
                words = []
                features = []
                chars = []
                labels = []
                objects = []
                word_Ids = []
                feature_Ids = []
                char_Ids = []
                label_Ids = []
                object_Ids = []
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
            instence_texts.append([words, features, chars, labels, objects])
            instence_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids, object_Ids])
            words = []
            features = []
            chars = []
            labels = []
            objects = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            label_Ids = []
            object_Ids = []
    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim

# ============= for vocab affix model ==============

def load_affix(filename):
    """
    Args:
        filename: suff.txt or pref.txt
    Returns:
        d: list[]，按词缀长度降序排序
    """
    d = set()
    with open(filename, 'r', encoding='utf-8') as f:
       for line in f.readlines():
            word = line.strip()
            d.add(word)
    d = list(d)
    d = sorted(d, key=lambda x: len(x), reverse=True)
    return d


def build_vocab_pref_suff_data(pref_file, suff_file, input_file, out_file, str=' '):
    """
    构建affix特征，按最长子串匹配
    :param affix_vocab_file: 词缀词表文件，(word, freq) per line, 用于过滤低词频的affix
    :param input_file: train,dev,test
    :param out_file: (text, pref, suff, tag) per line
    :return:
    """
    UNKNOWN = "</unk>"
    pref_vocab_lt = load_affix(pref_file)
    suff_vocab_lt = load_affix(suff_file)

    f_out = open(out_file,'w', encoding='utf-8')
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line != "":
                span_list = line.split(str)
                word = ''.join(list(span_list[0]))

                # 过滤掉低词频的词
                word_pref, word_suff = UNKNOWN, UNKNOWN
                # find pref
                for pref in pref_vocab_lt:
                    if word.startswith(pref):
                        word_pref = pref
                        break

                # find suff
                for suff in suff_vocab_lt:
                    if word.endswith(suff):
                        word_suff = suff
                        break

                tag = span_list[-1]
                f_out.write(' '.join([word, '[Pref]'+word_pref, '[Suff]'+word_suff, tag]) + '\n')
            else:
                f_out.write('\n')
    f_out.close()

# ============= for deep affix model ==============

def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: list of tuple(word, freq)
        filename: path to vocab file
    Returns:
        write a 'word freq' per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, (word, freq) in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{} {}\n".format(word, freq))
            else:
                f.write("{} {}".format(word, freq))
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Args:
        filename: file with a (word, freq) per line
    Returns:
        d: dict[word] = freq
    """
    d = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            word, freq = line.strip().split()
            d[word] = freq
    return d


def get_pref_suff(word, n_gram=3):
    """
    only support n-n_gram=3
    :param word:
    :return:
    pref, suff
    """
    pref = ""
    suff = ""

    if len(word) >= n_gram:
        pref = word[:n_gram]
        suff = word[-n_gram:]
    else:
        add_str = ""
        for _ in range(n_gram - len(word)):
            add_str += '_'
        pref = add_str + str(word)
        suff = str(word) + add_str
    return pref, suff


def build_pref_suff_vocab(number_normalized, train_file, vocab_file, n, T):
    """
        统计word的前后3-gram且在训练集中出现次数大于50的affix，并写入文件
    :param file: train data
    :return:
    """
    print('Build pref_suff vocab, with {}-gram, thresdhold: {}, normalize_word: True...'.format(n, T))
    pref_suff_vocab = {}
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 2:
                word_label = line.strip().split()
                word = word_label[0]
                if number_normalized:
                    word = normalize_word(word)
                pref, suff = get_pref_suff(word, n_gram=n)
                pref_suff_vocab[pref] = pref_suff_vocab.get(pref, 0) + 1
                pref_suff_vocab[suff] = pref_suff_vocab.get(suff, 0) + 1
    # filter
    pref_suff_vocab = [(word, freq) for word, freq in pref_suff_vocab.items() if freq >= T]
    # sort
    # -> [(word, freq),]
    pref_suff_vocab = sorted(pref_suff_vocab, key=lambda x: x[1], reverse=True)
    write_vocab(pref_suff_vocab, vocab_file)


def build_pref_suff_data(affix_vocab_file, input_file, out_file, number_normalized=True):
    """
    构建affix特征
    :param affix_vocab_file: 词缀词表文件，(word, freq) per line, 用于过滤低词频的affix
    :param input_file: train,dev,test
    :param out_file: (text, pref, suff, tag) per line
    :return:
    """
    UNKNOWN = "</unk>"
    affix_vocab = load_vocab(affix_vocab_file)
    f_out = open(out_file, 'w', encoding='utf-8')
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                span_list = line.split(' ')
                word = ''.join(list(span_list[0]))
                if number_normalized:
                    word_nor = normalize_word(word)
                # 过滤掉低词频的词
                pref, suff = get_pref_suff(word_nor, n_gram=3)
                if pref not in affix_vocab.keys():
                    pref = UNKNOWN
                if suff not in affix_vocab.keys():
                    suff = UNKNOWN
                tag = span_list[-1]
                f_out.write(' '.join([word, '[Pref]' + pref, '[Suff]' + suff, tag]) + '\n')
            else:
                f_out.write('\n')
    f_out.close()


if __name__ == '__main__':

    #============== vocab affix model ========================
    # build_vocab_pref_suff_data(pref_file='../data/affix/pref2.txt',
    #                      suff_file='../data/affix/suff4.txt',
    #                      input_file='../data/tweets_wnut2016/dev.BIO',
    #                      out_file='../data/tweets_wnut2016/dev.affix4_2.BIO')
    #
    # build_vocab_pref_suff_data(pref_file='../data/affix/pref2.txt',
    #                      suff_file='../data/affix/suff4.txt',
    #                      input_file='../data/tweets_wnut2016/test.BIO',
    #                      out_file='../data/tweets_wnut2016/test.affix4_2.BIO')
    #
    # build_vocab_pref_suff_data(pref_file='../data/affix/pref2.txt',
    #                      suff_file='../data/affix/suff4.txt',
    #                      input_file='../data/tweets_wnut2016/train.BIO',
    #                      out_file='../data/tweets_wnut2016/train.affix4_2.BIO')


    #============== deep affix model ======================
    # build_pref_suff_vocab(number_normalized=True, train_file='../data/tweets_wnut2016/train.BIO',
    #                       vocab_file='../data/tweets_wnut2016/pref_suff_vocab_3_75.txt',n=3, T=75)
    # build_pref_suff_data(affix_vocab_file='../data/tweets_wnut2016/pref_suff_vocab_3_75.txt',
    #                      input_file='../data/tweets_wnut2016/train.BIO',
    #                      out_file='../data/tweets_wnut2016/train.deepAffix_3_75.BIO', number_normalized=True)
    #
    # build_pref_suff_data(affix_vocab_file='../data/tweets_wnut2016/pref_suff_vocab_3_75.txt',
    #                      input_file='../data/tweets_wnut2016/dev.BIO',
    #                      out_file='../data/tweets_wnut2016/dev.deepAffix_3_75.BIO', number_normalized=True)
    #
    # build_pref_suff_data(affix_vocab_file='../data/tweets_wnut2016/pref_suff_vocab_3_75.txt',
    #                      input_file='../data/tweets_wnut2016/test.BIO',
    #                      out_file='../data/tweets_wnut2016/test.deepAffix_3_75.BIO', number_normalized=True)


    # # #============== CoNLL 2003 eng vocab affix ======================
    # build_vocab_pref_suff_data(pref_file='../data/affix/pref.txt',
    #                      suff_file='../data/affix/suff.txt',
    #                      input_file='../data/CoNLL2003/eng.testa.BIO',
    #                      out_file='../data/CoNLL2003/eng.testa.affix.BIO')
    #
    # build_vocab_pref_suff_data(pref_file='../data/affix/pref.txt',
    #                      suff_file='../data/affix/suff.txt',
    #                      input_file='../data/CoNLL2003/eng.testb.BIO',
    #                      out_file='../data/CoNLL2003/eng.testb.affix.BIO')
    #
    # build_vocab_pref_suff_data(pref_file='../data/affix/pref.txt',
    #                      suff_file='../data/affix/suff.txt',
    #                      input_file='../data/CoNLL2003/eng.train.BIO',
    #                      out_file='../data/CoNLL2003/eng.train.affix.BIO')
    #
    # # #============== CoNLL 2003 eng deep affix model ======================
    # build_pref_suff_vocab(number_normalized=True, train_file='../data/CoNLL2003/eng.train.BIO', vocab_file='../data/CoNLL2003/pref_suff_vocab.txt')
    # build_pref_suff_data(affix_vocab_file='../data/CoNLL2003/pref_suff_vocab.txt',
    #                      input_file='../data/CoNLL2003/eng.train.BIO',
    #                      out_file='../data/CoNLL2003/eng.train.deepAffix.BIO', number_normalized=True)
    #
    # build_pref_suff_data(affix_vocab_file='../data/CoNLL2003/pref_suff_vocab.txt',
    #                      input_file='../data/CoNLL2003/eng.testa.BIO',
    #                      out_file='../data/CoNLL2003/eng.testa.deepAffix.BIO', number_normalized=True)
    #
    # build_pref_suff_data(affix_vocab_file='../data/CoNLL2003/pref_suff_vocab.txt',
    #                      input_file='../data/CoNLL2003/eng.testb.BIO',
    #                      out_file='../data/CoNLL2003/eng.testb.deepAffix.BIO', number_normalized=True)

    build_vocab_pref_suff_data(pref_file='../data/affix/pref.txt',
                         suff_file='../data/affix/suff4.txt',
                         input_file='../data/tweets_new/test',
                         out_file='../data/tweets_new/test.affix', str='	')