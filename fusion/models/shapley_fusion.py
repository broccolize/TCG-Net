from dataclasses import dataclass # for FusionOutput
import torch
import torch.nn as nn
import torch.nn.functional as F # this is for softmax()

@dataclass
class FusionOutput:
    # 4.1
    y_hat_f: torch.Tensor # [B]
    y_hat_p: torch.Tensor # [B]
    y_hat_fp: torch.Tensor # [B]
    # 4.2 
    phi_f: torch.Tensor # [B]
    phi_p: torch.Tensor # [B]
    phi_int: torch.Tensor # [B]
    #4.3
    z_fusion: torch.Tensor # [B, 128]
    w: torch.Tensor # [B, 3]

class CooperativeShapleyFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, joint_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.H_f = nn.Linear(embed_dim, 1) #128 feature 1 output -> 1 subject
        self.H_p = nn.Linear(embed_dim, 1)
        # this is just the architecture, the real zjoint is compute later
        self.H_fp = nn.Sequential(nn.Linear(embed_dim*4, joint_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(joint_hidden, 1) )

        self.register_buffer('y_baseline', torch.tensor(0.0))
        self.W_f = nn.Linear(embed_dim, embed_dim, bias = False)
        self.W_p = nn.Linear(embed_dim, embed_dim, bias = False)
        self.W_int = nn.Linear(embed_dim, embed_dim, bias = False)


    def set_baseline(self, y_mean: float):
        self.y_baseline.fill_(y_mean)
        

    #this is only a foward pass so value are discarded after each calculation
    def forward(self, z_f: torch.Tensor, z_p: torch.Tensor,):
        y_hat_f = self.H_f(z_f).squeeze(-1) # [B, 1] -> squeeze -> [B]
        y_hat_p = self.H_p(z_p).squeeze(-1)
        z_joint = torch.cat([z_f, z_p, z_f * z_p, (z_f - z_p).abs()], dim = 1)
        y_hat_fp = self.H_fp(z_joint).squeeze(-1)
        phi_f = 0.5 * ((y_hat_f - self.y_baseline) + (y_hat_fp - y_hat_p))
        phi_p = 0.5 * ((y_hat_p - self.y_baseline) + (y_hat_fp - y_hat_f))
        phi_int = y_hat_fp - self.y_baseline - phi_f - phi_p

        w = F.softmax(torch.stack([phi_f, phi_p, phi_int], dim = 1), dim = 1)
        w_f, w_p, w_int = w[:, 0:1], w[:, 1:2], w[:, 2:3]
        z_prime_f = self.W_f(z_f)
        z_prime_p = self.W_p(z_p)
        z_prime_int = self.W_int(z_f * z_p)
        z_fusion = w_f * z_prime_f  +  w_p * z_prime_p  +  w_int * z_prime_int

        return FusionOutput(
            y_hat_f = y_hat_f,
            y_hat_p = y_hat_p,
            y_hat_fp = y_hat_fp,
            phi_f = phi_f, # 4.2
            phi_p = phi_p,
            phi_int = phi_int,
            z_fusion = z_fusion, # 4.3
            w = w,

        )


