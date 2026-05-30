import torch
import torch.nn as nn
import torch.nn.functional as F
from models.shapley_fusion import FusionOutput

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
       
       w_l1 = fusion.w.abs().mean()
       sp_l1 = s_p.abs().mean()
       af_l1 = alpha_f.abs().mean()

       l_sparse = w_l1 + sp_l1 + af_l1
       l_total = l_main + self.lambda_aux * l_aux + self.lambda_shapley * l_shapley + self.lambda_sparse * l_sparse

       return {
        "total":      l_total,    # call .backward() on this
        "l_main":     l_main,
        "l_aux":      l_aux,
        "l_shapley":  l_shapley,
        "l_sparse":   l_sparse,
        "w_l1":       w_l1,       # for logging
        "sp_l1":      sp_l1,      # for logging
        "af_l1":      af_l1,      # for logging
    }



