import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config


def min_max_normalize(sims):
    """
    Min-max normalize the similarity scores to [0, 1]
    """
    sims_min = sims.min()
    sims_max = sims.max()
    normalized_sims = (sims - sims_min) / (sims_max - sims_min + 1e-8)
    return normalized_sims


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
    仅支持“一对多”文本→视频检索。
    输入：已算好的相似度矩阵 S 和 Bool 匹配矩阵 match
    """

    def __init__(self, gamma=32, m=0.25):
        super().__init__()
        self.gamma = gamma
        self.m = m
        self.Delta_p = 1 - m
        self.Delta_n = m
        print("Using Circle Loss")

    def forward(self, sims, label):
        """
        S:     (B, B)  相似度矩阵，已归一化
        match: (B, B)  Bool 张量，True 表示正例
        return: 标量 loss
        """
        # 保证是 Bool 类型
        mask_p = label.bool()
        mask_n = ~mask_p

        # 论文式 (2) 自适应权重 α
        alpha_p = (self.Delta_p - sims).relu() * mask_p.float()
        alpha_n = (sims - self.Delta_n).relu() * mask_n.float()

        # 论文式 (4) 指数项
        exp_pos = (-self.gamma * alpha_p * (sims - self.Delta_p)).exp() * mask_p.float()
        exp_neg = (self.gamma * alpha_n * (sims - self.Delta_n)).exp() * mask_n.float()

        # 论文式 (4) 先求和再相乘
        L_p = exp_pos.sum(dim=1)  # 每行文本对自己的所有正例求和
        L_n = exp_neg.sum(dim=1)  # 每行文本对自己的所有负例求和
        product = (L_p * L_n).clamp(min=1e-8, max=1e8)

        loss = (1 + product).log().mean()
        return loss


class MixLoss(nn.Module):
    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha
        self.clip_loss_fn = CLIPLoss()
        self.circle_loss_fn = CircleLoss()

    def forward(self, sims, pos_mask, logit_scale):
        clip_loss = self.clip_loss_fn(sims, logit_scale)
        circle_loss = self.circle_loss_fn(sims, pos_mask)
        return self.alpha * clip_loss + (1 - self.alpha) * circle_loss


class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == "clip":
            return CLIPLoss()
        elif config.loss == "circle":
            return CircleLoss()
        elif config.loss == "mix":
            return MixLoss()
        else:
            raise NotImplementedError(f"Loss {config.loss} not implemented.")
