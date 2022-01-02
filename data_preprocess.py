import os
import json
import codecs
import regex
from xpinyin import Pinyin
from collections import Counter
from itertools import chain

import config

'''
    数据预处理
    输入格式：按照sample.txt文件写，一行一句话
    输出：data.txt（拼音序列\t汉字序列），vocab.json
'''

'''
    通过汉字句子得到拼音序列
    :param sentences 汉字句子
'''
def align(sentences):
    pinyin = Pinyin()
    pylist = pinyin.get_pinyin(sentences, " ").split()

    hanzilist = []
    for char, p in zip(sentences.replace(" ", ""), pylist):
        hanzilist.extend([char] + ["_"] * (len(p) - 1))

    py_str = "".join(pylist)
    hanzi_str = "".join(hanzilist)

    assert len(py_str) == len(hanzi_str), "The hanzis and the pinyins must be the same in length."
    return py_str, hanzi_str

'''
    用正则表达式过滤掉字母、阿拉伯数字等
'''
def clean(text):
    if regex.search("[A-Za-z0-9]", text) is not None: # For simplicity, roman alphanumeric characters are removed.
        return ""
    text = regex.sub(u"[^ \p{Han}。，！？]", "", text)
    return text

'''
    构建词表
    :param hanzo_sent_list 训练语料库中的所有句子
'''
def build_vocab(hanzi_sent_list):
    # 拼音部分
    if config.is_qwerty:
        pinyins = "EUabcdefghijklmnopqrstuvwxyz0123456789。，！？"  # E: Empty, U: Unknown
        pinyin2idx = {pnyn: idx for idx, pnyn in enumerate(pinyins)}
        idx2pinyin = {idx: pnyn for idx, pnyn in enumerate(pinyins)}
    else:
        pinyin2idx, idx2pinyin = dict(), dict()
        pnyns_list = ["E", "U", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz",
                      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", u"。", u"，", u"！",
                      u"？"]  # E: Empty, U: Unknown
        for i, pinyins in enumerate(pnyns_list):
            for pnyn in pinyins:
                pinyin2idx[pnyn] = i

    # 汉字部分
    hanzi2cnt = Counter(chain.from_iterable(hanzi_sent_list))
    hanzis = [hanzi for hanzi, cnt in hanzi2cnt.items() if cnt > 5]    # 去掉长尾字符

    if "_" in hanzis:
        hanzis.remove("_")
    hanzis = ["E", "U", "_"] + hanzis  # 0: empty, 1: unknown, 2: blank
    hanzi2idx = {hanzi: idx for idx, hanzi in enumerate(hanzis)}
    idx2hanzi = {idx: hanzi for idx, hanzi in enumerate(hanzis)}

    return pinyin2idx, idx2pinyin, hanzi2idx, idx2hanzi


if __name__ == '__main__':
    hanzi_sent_list = []    # 用于构建汉字vocab
    fout = open("data/data.txt", 'w', encoding='utf-8')
    with codecs.open("data/sample.txt", 'r', 'utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break

            sent = line.strip()
            sent = clean(sent)
            hanzi_sent_list.append(sent)
            if len(sent) > 0:
                py_str, hanzi_str = align(sent)
                fout.write("%s\t%s\n" % (py_str, hanzi_str))
                fout.flush()

        # 保存文件
        fout.close()

        # 构建vocab
        pinyin2idx, idx2pinyin, hanzi2idx, idx2hanzi = build_vocab(hanzi_sent_list)
        if config.is_qwerty:
            json.dump((pinyin2idx, idx2pinyin, hanzi2idx, idx2hanzi), open('data/vocab.qwerty.json', 'w'))
        else:
            json.dump((pinyin2idx, idx2pinyin, hanzi2idx, idx2hanzi), open('data/vocab.nine.json', 'w'))

