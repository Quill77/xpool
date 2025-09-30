import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, sims):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
        """
        n = sims.size(0)
        diagonal = sims.diag().view(n, 1)
        d1 = diagonal.expand_as(sims)
        d2 = diagonal.t().expand_as(sims)

        cost_s = (self.margin + sims - d1).clamp(min=0)
        cost_im = (self.margin + sims - d2).clamp(min=0)

        mask = torch.eye(n, device=sims.device).bool()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        return cost_s.sum() + cost_im.sum()

class SoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, labels):
        """
        Inputs:
            sims: n x m tensor, where n is the number of texts and m is the number of videos.
            labels: n x m tensor, where labels[i, j] is 1 if text i matches video j, otherwise 0.
        """
        # Normalize the similarities to get probabilities
        logits = F.log_softmax(sims, dim=1)
        
        # Compute the loss
        loss = -torch.sum(labels * logits) / labels.sum()
        print("labels[0]:", labels[0], flush=True)
        print("logits[0]:", logits[0], flush=True)
        print("labels.shape:", labels.shape, flush=True)
        print("logits.shape:", logits.shape, flush=True)
        print("labels.sum():", labels.sum(), flush=True)
        return loss

class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return CLIPLoss()
        elif config.loss == 'contrastive':
            return ContrastiveLoss(margin=config.contrastive_margin)
        elif config.loss == 'softmax':
            return SoftmaxLoss()
        else:
            raise NotImplemented
