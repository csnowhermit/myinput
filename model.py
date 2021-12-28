import torch
import torch.nn as nn

import config

'''
    输入法模型
'''

class MyInput(nn.Module):
    def __init__(self, vocab_size, hanzi_size, word_dim=300, encoder_num_banks=16, num_highwaynet_blocks=4):
        super(MyInput, self).__init__()
        self.vocab_size = vocab_size    # 所有拼音数量
        self.hanzi_size = hanzi_size    # 所有汉字数量
        self.word_dim = word_dim    # embedding成多少维
        self.encoder_num_banks = encoder_num_banks    # encoder结构重复次数
        self.num_highwaynet_blocks = num_highwaynet_blocks    # 高层次特征残差块的个数
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=1)
        self.prenet = nn.Sequential(
            nn.Linear(word_dim, word_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(word_dim, word_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.encoder = []
        self.encoder.append(nn.Conv1d(word_dim//2, 150, 1, dilation=1, padding='SAME', bias=False))
        for k in range(2, encoder_num_banks + 1):
            self.encoder.append(nn.Conv1d(word_dim//2, 150, kernel_size=k, dilation=1, padding='SAME', bias=False))


    def forward(self, x):
        emb = self.embedding(x)    # 先做embedding
        preout = self.prenet(emb)    # 做encoder前的预处理

        # encoder部分
        num_units = self.embedding // 2
        outputs = nn.Conv1d(x, 150, 1, dilation=1, padding='SAME', bias=False)
        for k in range(2, self.encoder_num_banks+1):
            output = nn.Conv1d(x, num_units, k)
            outputs = torch.cat((outputs, output), axis=-1)    # -1表示按最后一个维度拼接
        outputs = self.batch_normalize(outputs)
        outputs = self.relu(outputs)

        # maxpooling
        outputs = nn.MaxPool2d(outputs, 2, 1, padding='same')
        outputs = nn.Conv1d(outputs, self.word_dim // 2, 5, padding='SAME')
        outputs = self.batch_normalize(outputs)
        outputs = self.relu(outputs)
        outputs = nn.Conv1d(outputs, self.word_dim // 2, 5, padding='SAME')
        outputs = self.batch_normalize(outputs)
        outputs = self.relu(outputs)

        outputs += preout    # 相当于残差连接

        # highway-network
        num_units = outputs.shape[-1]
        for i in range(self.num_highwaynet_blocks):
            H = self.relu(nn.Linear(outputs, num_units))
            T = self.sigmoid(nn.Linear(outputs, num_units))
            C = 1. - T
            outputs = H * T + outputs * C

        # 双向GRU结构
        gru = nn.GRU(150, 150, 2)
        input = torch.randn(200, 16, 150)
        h0 = torch.randn(2, 16, 150)
        outputs, hn = gru(outputs, h0)
        outputs = torch.cat(outputs, 2)

        ## Readout
        outputs = nn.Linear(outputs, self.hanzi_size, bias=False)
        return outputs

    def batch_normalize(self, x):
        x = torch.unsqueeze(x, dim=1)    # 先扩展为四维向量
        x = nn.BatchNorm2d(x)
        return torch.squeeze(x, dim=1)    # 再还原回三维向量

if __name__ == '__main__':
    myinput = MyInput(vocab_size=42, hanzi_size=400)    # 42个字母，400个汉字
    x = torch.randn([config.batch_size, 16, 300], dtype=torch.float32)
    outputs = myinput(x)
    outputs = torch.argmax(outputs)  # 找属于哪个汉字的得分值最大，这里拿到的是index


