# Pytorch 依賴
import torch
import torchaudio
from torch.utils.data import DataLoader
from wsj0_person_dataloader import WSJ0_N_MIX_BY_PERSON
from models.conv_tasnet import ConvTasNet

# 不同的 Loss, 搭配 PIT
# SDR, SI-SDR, SA-SDR, SI-SNR
from torchmetrics.functional.audio import signal_distortion_ratio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torchmetrics.functional.audio import permutation_invariant_training

# W & B 依賴
import wandb

# 內部依賴
import logging
logger = logging.getLogger(__name__)

"""
超參數
批次大小、混合聲音數量、聲音長度、訓練次數、學習率、
是否啟用 W&B
"""
BATCH_SIZE       = 1
MIX_NUMBER       = 2
MIX_SOUND_LENGTH = 5
EPOCHS           = 10
LR               = 3e-4
WEIGHT_DECAY     = 1e-8
ENABLE_WANDB     = True
LOG_LEVEL        = logging.INFO


if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)

    """
    讀取資料集
    """
    wsj0_2mix_dataset = WSJ0_N_MIX_BY_PERSON(
        mix_number=MIX_NUMBER,
        mix_sound_length=MIX_SOUND_LENGTH
    )
    wsj0_2mix_dataloader = DataLoader(
        wsj0_2mix_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    
    """
    模型初始化設定模型的參數
    設定優化器
    """
    model = ConvTasNet(
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
        msk_activate         = "relu",
    ).to(device="cuda").train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


    """
    如果打開則會將訓練數據上傳至 W&B
    """
    if ENABLE_WANDB:
        wandb.init(
            project='DeepLearningProject', 
            name=f'ConvTasNet')
        wandb.watch(model)


    """
    開始訓練
    """
    for epoch in range(EPOCHS):
        for index, (mix_sound, target) in enumerate(wsj0_2mix_dataloader):
            mix_sound = mix_sound.to(device="cuda")
            target    = target.to(device="cuda")

            optimizer.zero_grad()
            preds = model(mix_sound)

            """
            Loss計算 透過PIT找到最佳排序方式
            """
            best_metric, best_perm = permutation_invariant_training(
                preds=preds,
                target=target,
                mode="speaker-wise",
                metric_func=scale_invariant_signal_noise_ratio,
                eval_func="max"
            )
            best_metric = torch.negative(best_metric)

            best_metric.backward()
            optimizer.step()

            logger.info(f"Epoch: {epoch}/{EPOCHS}, Iter: {index}/{len(wsj0_2mix_dataloader)}")
            logger.info(f"SI-SNR: {-best_metric.item():.2f} db\n")
            if ENABLE_WANDB: wandb.log({"SI-SNR": -best_metric.item()})

            """
            保存模型訓練結果 (調適用)
            """
            if index % 3000 == 0:
                torchaudio.save(
                    f"./exp/{index}_mix_sound.wav",
                    mix_sound[0].cpu().detach(),
                    sample_rate=32000
                )
                torchaudio.save(
                    f"./exp/{index}_seperation1.wav",
                    preds[0][0].unsqueeze(0).cpu().detach(),
                    sample_rate=32000
                )
                torchaudio.save(
                    f"./exp/{index}_seperation2.wav",
                    preds[0][1].unsqueeze(0).cpu().detach(),
                    sample_rate=32000
                )
                torch.save(model.state_dict(), f"./exp/{index}_ConvTasNet.pt")