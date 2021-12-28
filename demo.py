import torch
import torch.nn as nn


gru = nn.GRU(150, 150, 2, bidirectional=True)
input = torch.randn(200, 16, 150)
h0 = torch.randn(4, 16, 150)
output, hn = gru(input, h0)
print(output.shape)
print(hn.shape)
