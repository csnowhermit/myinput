import torch
import torch.nn as nn

# batch_size=5000
# gru_layers = 2
# gru_hidden_dim = 16
# dim = 300
#
# gru = nn.GRU(dim, dim+1, gru_layers, bidirectional=True)
# h0 = torch.randn(gru_layers * 2, gru_hidden_dim, dim+1)
#
# input = torch.randn(batch_size, gru_hidden_dim, dim)
# output, hn = gru(input, h0)
# print(output.shape)
# print(hn.shape)


## embedding

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        word_dim = 300
        num_units = word_dim // 2
        self.embedding = nn.Embedding(42, 300, padding_idx=1)
        self.prenet = nn.Sequential(
            nn.Linear(word_dim, word_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(word_dim, num_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.conv = nn.Conv1d(in_channels=num_units, out_channels=num_units, kernel_size=1, dilation=1, bias=False)

    def forward(self, x):
        emb = self.embedding(x)    # emb后为：[batch_size, seqlen, emb_size]
        emb = self.prenet(emb)    # [3, 4, 150]
        emb = emb.permute(0, 2, 1)    # 调整通道顺序，改为：[batch_size, emb_size, seqlen]
        for i in range(10):
            y = self.conv(emb)
            print("y:", y.shape)
        return emb


'''
    Encoder部分
'''


class Encoder(nn.Module):
    def __init__(self, word_dim, encoder_num_banks):
        super(Encoder, self).__init__()
        self.word_dim = word_dim
        self.num_units = word_dim // 2
        self.encoder_num_banks = encoder_num_banks

        self.conv_block = []
        self.conv_block.append(
            nn.Conv1d(in_channels=self.num_units, out_channels=self.num_units, kernel_size=1, dilation=1, bias=False))
        for k in range(2, self.encoder_num_banks + 1):
            self.conv_block.append(nn.Conv1d(in_channels=self.num_units, out_channels=self.num_units, kernel_size=1, dilation=1))

        self.bn = self.batch_normalize
        self.relu = nn.ReLU()

        # maxpooling阶段
        self.maxpooling = nn.MaxPool1d(kernel_size=2, stride=1, padding='SAME'),
        self.mp_conv1d = nn.Conv1d(self.num_units, self.num_units // 2, kernel_size=5),

    def batch_normalize(self, x):
        x = torch.unsqueeze(x, dim=1)  # 先扩展为四维向量
        x = nn.BatchNorm2d(x)
        return torch.squeeze(x, dim=1)  # 再还原回三维向量

    def forward(self, x):
        # encoder
        print("x:", x.shape)
        outputs = self.conv_block[0](x)    # [3, 150, 4]
        for k in range(1, len(self.conv_block)):
            print("k:", k, x.shape, end=', ')
            tmp = self.conv_block[k]
            output = tmp(x)    # [3, 150, 3]
            print(output.shape)
            outputs = torch.cat([outputs, output], axis=-1)  # -1表示按最后一个维度拼接 [3, 150, 7]
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        # maxpooling
        outputs = self.maxpooling(outputs)
        outputs = self.mp_conv1d(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        outputs = self.mp_conv1d(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        return outputs

enc = Encoder(word_dim=300, encoder_num_banks=16)
print(enc)

x = [[1, 1, 2, 3],
     [2, 2, 3, 4],
     [3, 3, 4, 5]]
x = torch.LongTensor(x)
print(x.shape)

m = model()
y = m(x)    # 这里y已经是[batch_size, emb_size, seqlen]
print(y.shape)    # [3, 150, 4]

z = enc(y)
print(z.shape)

# # pool of size=3, stride=2
# m = nn.MaxPool1d(3, stride=1)
# input = torch.randn(20, 16, 50)
# output = m(input)
# print(output.shape)