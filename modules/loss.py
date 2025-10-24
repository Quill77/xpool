import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print("Using CLIP Loss")

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale

        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0


class CircleLoss(nn.Module):
    """
    仅优化文本→视频方向的多标签 Circle Loss
    """

    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma
        # 最优值与 margin
        self.O_p = 1 + m
        self.O_n = -m

    def forward(self, sims, pos_mask):
        """
        text_feat : [B, dim]  已归一化
        vid_feat  : [V, dim]  已归一化
        pos_vid_idx : list[list[int]]  长度=B
        """
        # 2. 构造正/负 mask
        pos_mask = pos_mask.bool()
        neg_mask = ~pos_mask

        # 3. 取出 sp 与 sn
        sp = sims[pos_mask]
        sn = sims[neg_mask]

        # 4. 计算自适应权重
        ap = torch.relu(self.O_p - sp)
        an = torch.relu(sn - self.O_n)

        # 5. Circle Loss 统一公式（只优化 t2v）
        logits = torch.cat([an * sn, -ap * sp], dim=0)
        loss = torch.logsumexp(self.gamma * logits, dim=0) / self.gamma
        return loss


class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == "clip":
            return CLIPLoss()
        elif config.loss == "circle":
            return CircleLoss()
        else:
            raise NotImplementedError(f"Loss {config.loss} not implemented.")
