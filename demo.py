import torch
import torch.nn as nn
import json


testDict = {}
testDict['a'] = 1
testDict['b'] = 2
testDict['c'] = 3

print(testDict)
print(testDict.get('d', 5))