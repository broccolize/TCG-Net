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

    #this is only a foward pass so value are discarded after each calculation
    def forward(self, z_f: torch.Tensor, z_p: torch.Tensor,):
        y_hat_f = self.H_f(z_f).squeeze(-1) # [B, 1] -> squeeze -> [B]
        y_hat_p = self.H_p(z_p).squeeze(-1)
        z_joint = torch.cat([z_f, z_p, z_f * z_p, (z_f - z_p).abs()], dim = 1)
        y_hat_fp = self.H_fp(z_joint).squeeze(-1)
        return FusionOutput(
            y_hat_f = y_hat_f,
            y_hat_p = y_hat_p,
            y_hat_fp = y_hat_fp,
            phi_f = None, # 4.2
            phi_p = None,
            phi_int = None,
            z_fusion = None, # 4.3
            w = None,

        )


if __name__ == "__main__":
    B = 8
    z_f = torch.randn(B, 128)
    z_p = torch.randn(B, 128)

    model = CooperativeShapleyFusion()
    out = model(z_f, z_p)

    assert out.y_hat_f.shape  == (B,), f"y_hat_f: {out.y_hat_f.shape}"
    assert out.y_hat_p.shape  == (B,), f"y_hat_p: {out.y_hat_p.shape}"
    assert out.y_hat_fp.shape == (B,), f"y_hat_fp: {out.y_hat_fp.shape}"
    print("§4.1 shapes OK")
