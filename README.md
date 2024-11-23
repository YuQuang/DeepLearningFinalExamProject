# ConvTasNet & Paraformer to separate speech & STT

使用`ConvTasNet`以及`Paraformer`實作的語音分割並轉成文字的模型


[ConvTasNet論文連結點我](https://arxiv.org/abs/1809.07454)


[Paraformer論文連結點我](https://arxiv.org/abs/2206.08317)

## 資料集 WSJ0-2mix
訓練ConvTasNet模型的資料採用`WSJ0`，訓練時取兩個人的音訊並混和，目標便是分離兩段單獨音訊。

[WSJ0資料來源點我](https://www.kaggle.com/datasets/sonishmaharjan555/wsj0-2mix)

## 如何訓練
1. 安裝環境
2. 下載資料集
2. 執行`python train_conv.py`

## 如何測試
1. 準備混和音訊
2. 執行`python test.py`