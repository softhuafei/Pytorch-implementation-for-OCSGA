# -*- coding: utf-8 -*-


from utils.tagSchemeConverter import *
from tqdm import  tqdm
from gensim.models import word2vec
import numpy as np

def get_CoNLL2003NER(input_file, output_file):
    """
        提取CoNLL2003数据中的NER部分
    :param input_file:
    :param output_file:
    :return: None
    """
    f_out = open(output_file, 'w', encoding='UTF-8')
    with open(input_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                span_list = line.split(' ')
                raw_char = ''.join(list(span_list[0]))
                tag = span_list[-1]
                f_out.write(' '.join([raw_char, tag]) + '\n')
            else:
                f_out.write('\n')
    f_out.close()


def removeSegmentation(input_file, output_file):
    """
    将带有segmentation信息的weibo NER转化为不带segmentation信息的NER数据，即标准的CONLL数据格式
    Sample:
    With segmentation:
    她0	O
    和0	O
    现0	B-PER.NOM
    任1	I-PER.NOM
    男0	B-PER.NOM
    友1	I-PER.NOM
    交0	O
    往1	O
    时0	O
    To without segmentation:
    她 O
    和 O
    现 B-PER.NOM
    任 I-PER.NOM
    男 B-PER.NOM
    友 I-PER.NOM
    交 O
    往 O
    时 O
    :param input_file:
    :param output_file:
    :return:
        output_file = 'simple_'+input_file
    """
    f_out = open(output_file, 'w', encoding='UTF-8')
    with open(input_file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                span_list = line.split('\t')
                raw_char = ''.join(list(span_list[0])[:-1])
                tag = span_list[-1]
                f_out.write(' '.join([raw_char, tag]) + '\n')
            else:
                f_out.write('\n')
    f_out.close()


# 处理tweet_new数据集的help function
def remove_imageID(input_file, output_file):
    """
    删除tweet_new中的IMAGEID
    :param input_file:
    :param output_file:
    :return:
    """
    output_file = open(output_file, 'w', encoding='utf-8')
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if line.startswith('IMGID'):
                continue
            else:
                output_file.write(line)
    output_file.close()


def load_word_matrix(vocabulary, size=200):
    """
        This function is used to convert words into word vectors
    """
    b = 0
    word_matrix = np.zeros((len(vocabulary)+1, size))
    model = word2vec.Word2Vec.load('../40Wtweet_200dim.model')
    for word, i in vocabulary.iteritems():
        try:
            word_matrix[i]=model[word.lower().encode('utf8')]
        except KeyError:
            # if a word is not include in the vocabulary, it's word embedding will be set by random.
            word_matrix[i] = np.random.uniform(-0.25,0.25,size)
            b+=1
    print('there are %d words not in model'%b)
    return word_matrix




def conver_genEmbed2formEmbed():
    model = word2vec.Word2Vec.load('../40Wtweet_200dim.model')




if __name__ == '__main__':
    import os

    curpath = os.path.abspath(os.curdir)
    print(curpath)

    # root = '../data/Weibo NER/'
    # weiboNER_files = ['weiboNER_2nd_conll.dev',
    #                   'weiboNER_2nd_conll.test',
    #                   'weiboNER_2nd_conll.train']
    #
    # for input_file in weiboNER_files:
    #     withoutSeg_file = 'simple_'+input_file
    #     removeSegmentation(root + input_file, root + withoutSeg_file)
    #     BIOES_file = 'BIOES' + input_file
    #     BIO2BIOES(root + withoutSeg_file, root + BIOES_file)
    # remove_imageID('./data/tweet_new/dev', './data/tweet_new/dev_without_IMAGEID')
    # remove_imageID('./data/tweet_new/train', './data/tweet_new/train_without_IMAGEID')
    # remove_imageID('./data/tweet_new/test', './data/tweet_new/test_without_IMAGEID')
    get_CoNLL2003NER('../data/CoNLL2003/eng.train', '../data/CoNLL2003/eng.train.BIO')
    get_CoNLL2003NER('../data/CoNLL2003/eng.testa', '../data/CoNLL2003/eng.testa.BIO')
    get_CoNLL2003NER('../data/CoNLL2003/eng.testb', '../data/CoNLL2003/eng.testb.BIO')
