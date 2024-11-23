# 內建依賴套件
import os
import glob
import itertools
import random

random.seed(10)

# 外部依賴套件
import torch
import torchaudio
from torch.utils.data import Dataset

class WSJ0_N_MIX_BY_PERSON(Dataset):
    def __init__(
            self,
            path="./wsj0_by_person",
            mix_number=2,
            sample_rate=16000,
            mix_sound_length=10,
            some_datasets=500000
        ):
        self.sample_rate = sample_rate
        self.mix_sound_length = mix_sound_length

        # 拿到資料夾底下所有音檔 並排列組合(不重複)
        self.all_person        = glob.glob(os.path.join(path, "*"))

        self.each_person_sound = []
        for each_person in self.all_person:
            self.each_person_sound.append(
                glob.glob(os.path.join(each_person, "*.wav"))
            )

        self.all_comb  = list(
                itertools.product(
                    self.each_person_sound[0], self.each_person_sound[1]
                )
            )

    def __getitem__(self, index):
        comb = self.all_comb[index]

        # 兩個說話的人聲, target為目標分離音訊
        sound1, sample_rate1 = torchaudio.load(comb[0])
        sound2, sample_rate2 = torchaudio.load(comb[1])

        # 混合後的聲音
        mix_length = len(sound1[0]) if len(sound1[0]) < len(sound2[0]) else len(sound2[0])
        mix_length = int(self.sample_rate*self.mix_sound_length) if mix_length > int(self.sample_rate*self.mix_sound_length) else mix_length
        trim_sound1 = sound1[0][:mix_length]
        trim_sound2 = sound2[0][:mix_length]

        mix_sound = trim_sound1 + trim_sound2
        mix_sound = mix_sound.unsqueeze(0)
        target = torch.cat((trim_sound1.unsqueeze(0), trim_sound2.unsqueeze(0)), 0)

        return mix_sound, target

    def __len__(self):
        return len(self.all_comb)