import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import parse_sentences
from model import MyInput

if __name__ == '__main__':
    # 读取词表
    if config.is_qwerty:
        pinyin2idx, idx2pinyin, hanzi2idx, idx2hanzi = json.load(open("data/vocab.qwerty.json", 'r', encoding='utf-8'))
    else:
        pinyin2idx, idx2pinyin, hanzi2idx, idx2hanzi = json.load(open("data/vocab.nine.json", 'r', encoding='utf-8'))

    myinput = MyInput(vocab_size=len(pinyin2idx.keys()), hanzi_size=len(hanzi2idx.keys()))
    myinput.to(config.device)
    if config.device.type == 'cuda':
        myinput.cuda()
    print(myinput)

    if len(config.pretrained_model) > 0:
        checkpoint = torch.load(config.pretrained_model, map_location=config.device)
        myinput.load_state_dict(checkpoint)
        print("Load pretrained model from %s" % config.pretrained_model)
    else:
        print("No pretrained model")
        exit()

    while True:
        line = input("输入：")    # 输入拼音序列
        pinyin_idxArr = parse_sentences(pinyin2idx, line)
        pinyin_idxArr = torch.LongTensor([pinyin_idxArr])
        outputs = myinput(pinyin_idxArr)    # [1, 50, 35]
        preds = torch.argmax(outputs, 2)    # 这里是生成汉字的序号

        # 汉字序列转为汉字文字，并替换E(空)和_(空格)
        result = "".join(idx2hanzi[str(idx)] for idx in preds.detach().cpu().numpy()[0]).replace("E", "").replace("_", "")
        print(result)
