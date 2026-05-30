import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.2, out_dim: int = 1 ):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(64,32), nn.GELU(),
                                 nn.Linear(32, out_dim))
        
    def forward(self, z_fusion: torch.Tensor):
        y_hat = self.net(z_fusion).squeeze(-1)
        return y_hat
    
