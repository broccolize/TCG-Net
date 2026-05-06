"""
Squeeze-and-Excitation (SE) block for the TCG-Net proteomics encoder.

Role in TCG-Net
---------------
The SE block is applied to the 1024-dimensional hidden representation
``h_p`` produced by Stage 1 of the ProteomicsEncoder (the first
Linear → BN → GELU → Dropout block). It learns a channel-wise gating
vector ``s_p`` that re-weights the hidden features before they are
passed further into the encoder.

Architecture
------------
    Input:  h_p  ∈ R^(B × 1024)
    W1:     Linear(1024, 256, bias=True)
    Act:    GELU
    W2:     Linear(256, 1024, bias=True)
    Gate:   Sigmoid
    s_p  =  Sigmoid(W2(GELU(W1(h_p))))   shape: (B, 1024)
    h̃_p =  h_p ⊙ s_p                    shape: (B, 1024)

Why s_p is returned explicitly
-------------------------------
``s_p`` must be accessible outside the block for three reasons:

1. **Sparsity regularisation** — Stage 1 loss adds
   ``lambda_sparse * ||s_p||_1`` (per-sample L1 mean, batch-averaged)
   to push the gate toward sparsity and improve latent selectivity.

2. **Interpretability** — the mean ``s_p`` across the dataset (a
   1024-d vector) is a direct readout of which latent channels are
   most active, used by the importance analysis script (Step 2.13).

3. **Logging** — ``sp_l1_norm`` is tracked in the per-epoch training
   CSV so gate collapse (all gates near 1) or gate death (all near 0)
   can be detected without loading a checkpoint.

Default dimensions match ``SE_REDUCTION_DIM = 256`` and
``HIDDEN_DIM_1 = 1024`` from ``config.py``.  Callers that need
non-default dimensions pass them explicitly to ``__init__``.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block that gates a 1024-d hidden representation.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input ``h_p`` (and the output gate ``s_p``).
        Defaults to 1024 to match ``HIDDEN_DIM_1`` in ``config.py``.
    reduction_dim : int
        Bottleneck width of the SE block's inner linear layer.
        Defaults to 256 to match ``SE_REDUCTION_DIM`` in ``config.py``.
    """

    def __init__(self, in_dim: int = 1024, reduction_dim: int = 256) -> None:
        super().__init__()
        # W1: squeeze hidden representation down to a bottleneck.
        self.W1 = nn.Linear(in_dim, reduction_dim, bias=True)
        # GELU activation (spec-mandated, not ReLU).
        self.act = nn.GELU()
        # W2: excite back to the full hidden dimension.
        self.W2 = nn.Linear(reduction_dim, in_dim, bias=True)
        # Sigmoid produces gate values in (0, 1) for each hidden unit.
        self.gate = nn.Sigmoid()

    def forward(
        self, h_p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gated hidden representation and the gate vector.

        Parameters
        ----------
        h_p : torch.Tensor, shape (B, in_dim)
            Hidden representation from Stage 1 of the ProteomicsEncoder.

        Returns
        -------
        h_tilde_p : torch.Tensor, shape (B, in_dim)
            Gated representation: ``h_p ⊙ s_p``.
        s_p : torch.Tensor, shape (B, in_dim)
            Per-sample gate vector; values in (0, 1). Used by the loss
            function for sparsity regularisation and by the importance
            analysis script.
        """
        # Squeeze: compress h_p to the bottleneck dimension.
        s_p = self.W1(h_p)          # (B, reduction_dim)
        s_p = self.act(s_p)         # GELU activation in the bottleneck
        # Excite: project back to the full hidden dimension.
        s_p = self.W2(s_p)          # (B, in_dim)
        # Gate: squash to (0, 1) so each value is a multiplicative weight.
        s_p = self.gate(s_p)        # (B, in_dim)

        # Element-wise gating: rescale each hidden unit by its gate value.
        h_tilde_p = h_p * s_p      # (B, in_dim)

        return h_tilde_p, s_p
