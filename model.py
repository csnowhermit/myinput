import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import myconv

'''
    输入法模型：emb + prenet + encoder + highway_network + gru
'''

class MyInput(nn.Module):
    def __init__(self, vocab_size, hanzi_size, word_dim=300, encoder_num_banks=16, num_highwaynet_blocks=4):
        super(MyInput, self).__init__()
        self.vocab_size = vocab_size    # 所有拼音数量
        self.hanzi_size = hanzi_size    # 所有汉字数量
        self.word_dim = word_dim    # embedding成多少维
        self.encoder_num_banks = encoder_num_banks    # encoder结构重复次数
        self.num_highwaynet_blocks = num_highwaynet_blocks    # 高层次特征残差块的个数

        num_units = self.word_dim // 2
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=1)
        self.prenet = nn.Sequential(
            nn.Linear(word_dim, word_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(word_dim, num_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # encoder部分
        self.encoder = Encoder(word_dim=self.word_dim, encoder_num_banks=16)

        # highway-network结构
        self.hn_linear = nn.Linear(num_units, num_units)

        # GRU
        self.gru_layers = 1    # GRU结构的层数
        self.gru_hidden_dim = config.maxlen    # GRU隐藏层的dim，等于一句话的最大长度
        self.gru = nn.GRU(num_units, num_units, self.gru_layers, bidirectional=True)    # 双向GRU：输入dim数，输出dim数，GRU结构的层数
        self.hidden = self._init_hidden(num_units)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # readout
        self.readout = nn.Linear(self.word_dim, self.hanzi_size, bias=False)


    def _init_hidden(self, dim):
        return torch.randn(self.gru_layers * 2, self.gru_hidden_dim, dim)    # 双向GRU的话隐藏层层数需*2，隐藏dim，输出dim（输出dim需与gru的输出保持一致）

    def forward(self, x):
        emb = self.embedding(x)    # 先做embedding， [3, 50, 300]
        preout = self.prenet(emb)    # 做encoder前的预处理, preout [3， 50， 150]

        preout = preout.permute(0, 2, 1)    # 调整通道顺序，改为：[batch_size, emb_size, seqlen] [3, 150, 50]
        # # encoder部分
        enc = self.encoder(preout)    # [3, 150, 50]
        enc += preout    # 残差连接 [3, 150, 50]

        # highway-network模块
        enc = enc.permute(0, 2, 1)    # 再转回[batch_size, seqlen, emb_size] [3, 50, 150]
        for i in range(self.num_highwaynet_blocks):
            H = self.relu(self.hn_linear(enc))
            T = self.sigmoid(self.hn_linear(enc))
            C = 1. - T
            enc = H * T + enc * C    # highway-network模块中，enc始终为 [3, 50, 150]

        # 双向GRU结构
        # print("GRU:", enc.shape)    # [3, 50, 150]
        outputs, hn = self.gru(enc, self.hidden)    # outputs [3, 50, 300]

        ## Readout
        outputs = self.readout(outputs)
        return outputs


'''
    Encoder部分
'''
class Encoder(nn.Module):
    def __init__(self, word_dim, encoder_num_banks):
        super(Encoder, self).__init__()
        self.word_dim = word_dim
        self.num_units = word_dim // 2
        self.encoder_num_banks = encoder_num_banks

        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=self.num_units, out_channels=self.num_units, kernel_size=1, dilation=1, bias=False))
        for k in range(2, self.encoder_num_banks + 1):
            # tensoeflow中使用padding='SAME'参数确保卷积前后矩阵大小相同。
            # 而pytorch是卷积前对边缘填充，padding=n，n表示在边缘填充几层。卷积之后的填充用F.pad()
            # conv_layers.append(nn.Conv1d(in_channels=self.num_units, out_channels=self.num_units, kernel_size=k))
            conv_layers.append(myconv.Conv1d(in_channels=self.num_units, out_channels=self.num_units, kernel_size=k))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.bn1 = nn.BatchNorm1d(self.encoder_num_banks * self.num_units)    # Conv1D projections之前用这个
        self.bn2 = nn.BatchNorm1d(self.num_units)    # Conv1D projections之后用这个bn
        self.relu = nn.ReLU()

        # maxpooling阶段
        self.maxpooling = nn.MaxPool1d(kernel_size=2, stride=1)    # maxpooling完了仍然需要填充至原来大小
        self.project1_conv1d = nn.Conv1d(self.encoder_num_banks * self.num_units, self.num_units, kernel_size=5)
        self.project2_conv1d = nn.Conv1d(self.num_units, self.num_units, kernel_size=5)

    def forward(self, x):
        # encoder
        outputs = self.conv_layers[0](x)    # x [3, 150, 50]
        for k in range(1, len(self.conv_layers)):
            output = self.conv_layers[k](x)

            # # 采用自定义Conv后就不用手动填充了
            # # 卷积之后填充至原来相同shape
            # padding_num = config.maxlen - output.shape[-1]
            # # padding = [math.floor(padding_num / 2), math.ceil(padding_num / 2)]    # 这种填充方式不对，应该在右或下填充
            # padding = [0, math.ceil(padding_num)]
            # output = F.pad(output, padding)

            outputs = torch.cat([outputs, output], axis=1)  # 在第二个维度拼接：emb_size维度

        outputs = self.bn1(outputs)    # outputs [3, 2400, 50]
        outputs = self.relu(outputs)

        # maxpooling
        outputs = self.maxpooling(outputs)    # [3, 2400, 49]
        padding_num = config.maxlen - outputs.shape[-1]
        padding = [math.floor(padding_num / 2), math.ceil(padding_num / 2)]
        outputs = F.pad(outputs, padding)    # 填充至原来大小 [3, 2400, 50]

        # Conv1D projections
        outputs = self.project1_conv1d(outputs)
        padding_num = config.maxlen - outputs.shape[-1]
        padding = [math.floor(padding_num / 2), math.ceil(padding_num / 2)]
        outputs = F.pad(outputs, padding)  # 填充至原来大小

        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)

        outputs = self.project2_conv1d(outputs)
        padding_num = config.maxlen - outputs.shape[-1]
        padding = [math.floor(padding_num / 2), math.ceil(padding_num / 2)]
        outputs = F.pad(outputs, padding)  # 填充至原来大小

        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)

        return outputs




if __name__ == '__main__':
    myinput = MyInput(vocab_size=42, hanzi_size=400)    # 42个字母，400个汉字
    x = [[1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [2, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [3, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    x = torch.LongTensor(x)
    outputs = myinput(x)
    print("outputs:", outputs.shape)    # [3, 50, 400], 400表示当前行当前字属于哪一个字的概率
    preds = torch.argmax(outputs, 2)  # 找属于哪个汉字的得分值最大，这里拿到的是index
    print("preds:", preds.shape)

