#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:46:22 2018
@author: lilong
"""

"""
由原始文本进行分词后保存到新文件
"""
import jieba
import numpy as np

filePath = '/root/train_data/corpus.txt'
fileSegWordDonePath = 'corpusSegDone_1.txt'


# 打印中文列表
def PrintListChinese(list):
    for i in range(len(list)):
        print(list[i])

    # 读取文件内容到列表


fileTrainRead = []
with open(filePath, 'r') as fileTrainRaw:
    for line in fileTrainRaw:  # 按行读取文件
        fileTrainRead.append(line)

# jieba分词后保存在列表中
fileTrainSeg = []
for i in range(len(fileTrainRead)):
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][9:-11], cut_all=False)))])
    if i % 100 == 0:
        print(i)

# 保存分词结果到文件中
with open(fileSegWordDonePath, 'w', encoding='utf-8') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0])
        fW.write('\n')

"""
gensim word2vec获取词向量
"""

import warnings
import logging
import os.path
import sys
import multiprocessing

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 忽略警告
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])  # 读取当前文件的文件名
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # inp为输入语料, outp1为输出模型, outp2为vector格式的模型
    inp = 'corpusSegDone_1.txt'
    out_model = 'corpusSegDone_1.model'
    out_vector = 'corpusSegDone_1.vector'

    # 训练skip-gram模型
    model = Word2Vec(LineSentence(inp), size=50, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # 保存模型
    model.save(out_model)
    # 保存词向量
    model.wv.save_word2vec_format(out_vector, binary=False)



