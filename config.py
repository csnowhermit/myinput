import os
import torch

'''
    配置项
'''

batch_size = 32
maxlen = 50    # 每个句子的最大长度
is_qwerty = True    # 26字母键盘，False为九宫格键盘
pretrained_model = "./checkpoint/myinput_29_0.6235_0.7344.pth"    # 预训练模型
# pretrained_model = ""

use_gpu = False    # 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
learning_rate = 0.0001
total_epochs = 30
