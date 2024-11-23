import torch
import torchaudio
from funasr import AutoModel

from models.conv_tasnet import ConvTasNet

"""
Load ConvTasNet Model for Speech Separation
"""
sep_model = ConvTasNet(
    num_sources          = 2,
    # encoder/decoder parameters
    enc_kernel_size      = 16,
    enc_num_feats        = 512,
    # mask generator parameters
    msk_kernel_size      = 3,
    msk_num_feats        = 128,
    msk_num_hidden_feats = 512,
    msk_num_layers       = 8,
    msk_num_stacks       = 3,
    msk_activate         = "relu"
)
sep_model.load_state_dict(torch.load("./exp/111000_ConvTasNet.pt", weights_only=True))
sep_model.eval()

"""
Load Paraformer Model for ASR
"""
asr_model = AutoModel(
    model="funasr/paraformer-zh",
    model_revision="v2.0.4",
    hub="hf",
    disable_update=True
)

"""
Load mixed sound
"""
mix_sound, _ = torchaudio.load("./mixed_sound.wav")
y = sep_model(mix_sound.unsqueeze(0))
print(y.shape)
torchaudio.save(
    f"./seperation1.wav",
    y[0][0].unsqueeze(0).cpu().detach(),
    sample_rate=32000
)
torchaudio.save(
    f"./seperation2.wav",
    y[0][1].unsqueeze(0).cpu().detach(),
    sample_rate=32000
)

"""

"""
output = asr_model.generate("seperation1.wav")
print(output)
output = asr_model.generate("seperation2.wav")
print(output)