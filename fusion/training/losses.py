import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion.models.shapley_fusion import FusionOutput

class Stage2Loss(nn.Module):
    def __init__(self, lambda_aux: float = 0.3, lambda_shapley: float = 0.1, lambda_sparse: float = 1e-4 ):
        super().__init__()

        self.lambda_aux = lambda_aux
        self.lambda_shapley = lambda_shapley
        self.lambda_sparse = lambda_sparse

        self.loss_fn = nn.HuberLoss(delta=1.0) # could also use MSE for later

    def forward(self, y_true, y_hat, fusion: FusionOutput, s_p, alpha_f, fusion_model):
       y0 = fusion_model.y_baseline
       l_main = self.loss_fn(y_hat, y_true)
       l_aux = F.mse_loss(fusion.y_hat_f, y_true) + F.mse_loss(fusion.y_hat_p, y_true) + F.mse_loss(fusion.y_hat_fp, y_true)
       l_shapley = (fusion.y_hat_fp - (fusion.phi_f + fusion.phi_p + fusion.phi_int + y0)).pow(2).mean()
       
       sp_l1 = s_p.abs().mean()   # logged for monitoring; not in l_sparse (suppresses SE gate)
       af_l1 = alpha_f.abs().mean()

       # DEVIATION from §6.4: uses .mean() not .sum() for size-invariance across tensors
       # of different shapes (w[B,3], s_p[B,1024], alpha_f[E,heads]).
       # w and sp_l1 removed (Session 3): w has zero gradient (softmax constant); sp_l1
       # suppresses SE gate toward 0, counterproductive for interpretability.
       l_sparse = af_l1
       l_total = l_main + self.lambda_aux * l_aux + self.lambda_shapley * l_shapley + self.lambda_sparse * l_sparse

       return {
        "total":      l_total,
        "l_main":     l_main,
        "l_aux":      l_aux,
        "l_shapley":  l_shapley,   # always 0 by construction (efficiency identity)
        "l_sparse":   l_sparse,
        "w_f":        fusion.w[:, 0].mean(),   # mean fMRI weight across batch
        "w_p":        fusion.w[:, 1].mean(),   # mean proteomics weight
        "w_int":      fusion.w[:, 2].mean(),   # mean interaction weight
        "sp_l1":      sp_l1,
        "af_l1":      af_l1,
    }



