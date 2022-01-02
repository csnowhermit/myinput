import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import split_dataset, Input_Dataset
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
        print("[!] Retrain")

    trainlist, vallist = split_dataset("data/data.txt")

    train_dataloader = torch.utils.data.DataLoader(Input_Dataset(trainlist, pinyin2idx, hanzi2idx), batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(Input_Dataset(vallist, pinyin2idx, hanzi2idx), batch_size=config.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()    # 分类问题
    optimizer = torch.optim.Adam(myinput.parameters(), lr=config.learning_rate)

    train_loss = 99999  # 记录当前train loss
    eval_loss = 99999  # 记录当前eval loss，只有当val集上的loss下降时才保存模型

    train_loss_list = []
    val_loss_list = []
    for epoch in range(config.total_epochs):
        start = time.time()
        myinput.train()
        for i, (pinyin_idxArr, hanzi_idxArr) in enumerate(train_dataloader):
            pinyin_idxArr = pinyin_idxArr.to(config.device, dtype=torch.long)  # [16, 1, 50]
            hanzi_idxArr = hanzi_idxArr.to(config.device, dtype=torch.long)  # [16, 1, 50]
            if config.device.type == 'cuda':
                pinyin_idxArr.cuda()
                hanzi_idxArr.cuda()

            optimizer.zero_grad()
            y_ = myinput(pinyin_idxArr)    # y_ [batch_size, maxlen, len(hanzi2idx.keys())]
            y_ = y_.permute(0, 2, 1)    # 计算损失时要转为 [16, 35, 50]
            loss = criterion(y_, hanzi_idxArr)
            print("Epoch: [%d/%d], Batch: [%d/%d], train_loss: %.6f, time: %.4f" % (epoch, config.total_epochs, i, len(train_dataloader), loss, time.time() - start))
            start = time.time()
            loss.backward()
            optimizer.step()
        curr_train_loss = loss.detach().cpu().numpy()
        train_loss_list.append(curr_train_loss)
        # 按训练集loss的更新保存
        if curr_train_loss < train_loss:
            train_loss = curr_train_loss  # 更新保存的loss

        myinput.eval()
        eval_losses = []  # 统计验证集的损失
        val_start = time.time()
        for i, (pinyin_idxArr, hanzi_idxArr) in enumerate(val_dataloader):
            pinyin_idxArr = pinyin_idxArr.to(config.device, dtype=torch.long)  # [16, 1, 50]
            hanzi_idxArr = hanzi_idxArr.to(config.device, dtype=torch.long)  # [16, 1, 50]

            optimizer.zero_grad()
            y_ = myinput(pinyin_idxArr)
            y_ = y_.permute(0, 2, 1)  # 计算损失时要转为 [16, 35, 50]
            loss = criterion(y_, hanzi_idxArr)
            eval_losses.append(loss.detach().cpu().numpy())
        curr_eval_loss = np.mean(eval_losses)
        val_loss_list.append(curr_eval_loss)  # 验证集的损失用平均损失
        print("Epoch: %d, Batch: %d, train_loss: %.6f, eval_loss: %.6f, time: %.4f" % (epoch, len(train_dataloader), curr_train_loss, curr_eval_loss, time.time() - val_start))

        # 保存模型
        if curr_train_loss < eval_loss:
            eval_loss = curr_eval_loss  # 更新保存的loss

            torch.save(myinput.state_dict(), "./checkpoint/myinput_%d_%.4f_%.4f.pth" % (epoch, train_loss, eval_loss))

        if epoch >= 0:
            plt.figure()
            plt.subplot(121)
            plt.plot(np.arange(0, len(train_loss_list)), train_loss_list)
            plt.subplot(122)
            plt.plot(np.arange(0, len(val_loss_list)), val_loss_list)
            plt.savefig("metrics.jpg")
            plt.close("all")
