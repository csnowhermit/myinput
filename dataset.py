import os
import json
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import config

'''
    拆分数据集
'''
def split_dataset(datapath):
    trainlist, vallist = [], []
    with open(datapath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if random.randint(0, 10) <= 8:
                trainlist.append(line.strip("\n"))
            else:
                vallist.append(line.strip("\n"))
    return trainlist, vallist

'''
    解析字符串，字符串转idx序列
    :param idxDict pinyin2idx或hanzi2idx
    :param sentences 
'''
def parse_sentences(idxDict, sentences):
    x = [idxDict.get(s, 1) for s in sentences]
    if len(x) > config.maxlen:
        x = x[:config.maxlen]
    else:
        x += [0] * (config.maxlen - len(x))    # 填充至最大长度
    X = np.array(x, np.long)
    return X

class Input_Dataset(Dataset):
    def __init__(self, datalist, pinyin2idx, hanzi2idx):
        self.datalist = datalist
        self.pinyin2idx = pinyin2idx    # 拼音和id的对应
        self.hanzi2idx = hanzi2idx    # 汉字和id的对应

    def __getitem__(self, index):
        pinyin_str, hanzi_str = self.datalist[index].split("\t")

        pinyin_idx = parse_sentences(self.pinyin2idx, pinyin_str)
        hanzi_idx = parse_sentences(self.hanzi2idx, hanzi_str)

        return pinyin_idx, hanzi_idx

    def __len__(self):
        return len(self.datalist)

if __name__ == '__main__':
    trainlist, vallist = split_dataset("data/data.txt")
    pinyin2idx, idx2pinyin, hanzi2idx, idx2hanzi = json.load(open("data/vocab.qwerty.json", 'r', encoding='utf-8'))

    train_dataloader = torch.utils.data.DataLoader(Input_Dataset(trainlist, pinyin2idx, hanzi2idx), batch_size=16, shuffle=True)
    for i, (pinyin_str, hanzi_str) in enumerate(train_dataloader):
        print(type(pinyin_str), pinyin_str.shape)
        pinyin_str.to(config.device, dtype=torch.long)