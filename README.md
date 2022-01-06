# myinput

基于encoder-decoder的拼音-中文输入法

# 1、训练

## 1.1、数据格式

​	数据集做成文本文件即可，格式参照data/sample.txt文件。

## 1.2、数据预处理

``` python
python data_preprocess.py
```

​	脚本运行完成后会在data/目录下生成两个文件：data.txt和vocab.qwerty.json

data.txt文件格式：拼音序列\t汉字序列，例：

``` python
wohaoxiangfaxianleyangchugaoshilimaodemijueai	我_好__像____发_现___了_养___出__高__势__力_猫__的_秘_诀__诶_
nichulemainaxiejinbidehuanmaileshenme	你_除__了_买__那_些__金__币_的_还___买__了_什___么_
huozhenimeitiancanjiacaiyidasai	或__者__你_每__天___参__加__才__艺_大_赛__
dajiachongyixiama	大_家__冲____一_下__嘛_
niyaoshibuxiang，nashuixianzaizhedaodaode	你_要__是__不_想____，那_谁___先___在__这__叨__叨__的_
womenduzaizhongwushijiankaifubende	我_们__都_在__中____午_时__间___开__副_本__的_
```

vocab.qwerty.json文件包含四个Dict：pinyin2idx, idx2pinyin, hanzi2idx, idx2hanzi。

分别表示拼音和idx互相对应关系，汉字和idx互相对应关系。

## 1.3、训练

``` python
python train.py
```

​	训练好的模型保存在checkpoint/目录下。

# 2、测试

1、先修改config.py文件：pretrained_model项改为要测试的模型路径

``` python
pretrained_model = "./checkpoint/myinput_29_0.6529_0.7107.pth"    # 预训练模型
```

2、运行测试脚本：

``` python
pythonn test.py
```

Demo：

``` python
Load pretrained model from ./checkpoint/myinput_29_0.6529_0.7107.pth
输入：chuangqianmingyueguang
窗前明月光
输入：yishidishangshuang
疑是地上霜
输入：jutouwagnmingyue
举头望明月
输入：ditousiguxiang
低头思故乡
```

