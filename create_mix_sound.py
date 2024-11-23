from torch import Tensor
import torch.nn.functional as F
import torchaudio


def trim_sound_length(sound: Tensor) -> Tensor:
    if sound.size(1) < TARGET_LENGTH:
        pad_length = TARGET_LENGTH - sound.size(1)
        pad = (0, pad_length)
        padded_tensor = F.pad(sound, pad, mode='constant', value=0)
    else:
        padded_tensor = sound[:TARGET_LENGTH]
    print(padded_tensor.shape)
    return padded_tensor


"""
設定產生出來的音訊資訊
"""
SAMPLE_RATE   = 32000
MAX_LENGTH    = 10
TARGET_LENGTH = SAMPLE_RATE * MAX_LENGTH

"""
讀取兩段音訊
"""
sound1, sample_rate1 = torchaudio.load("wsj0_by_person/440/440a010a.wav")
sound2, sample_rate2 = torchaudio.load("wsj0_by_person/441/441a010a.wav")


"""
開始混合
"""
sound1 = trim_sound_length(sound1)
sound2 = trim_sound_length(sound2)
mix_sound = sound1 + sound2
torchaudio.save("mixed_sound.wav", src=mix_sound, sample_rate=SAMPLE_RATE)