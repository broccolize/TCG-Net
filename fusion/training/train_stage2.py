import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fmri" / "Code"))

from train_fmri_stage1 import (
    ConnectomeEncoder, build_group_mask, compute_graph_features, ConnectomeDataset,
)
from fusion.training.losses import Stage2Loss
from proteomics.models.encoder import ProteomicsEncoder
from fusion.models import FusionOutput, CooperativeShapleyFusion, PredictionHead

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_fc(x_f: np.ndarray) -> np.ndarray:
    """Upper-triangle vectors [N, 2145] → symmetric FC matrices [N, 66, 66]."""
    N = x_f.shape[0]
    fc = np.zeros((N, 66, 66), dtype=np.float32)
    rows, cols = np.triu_indices(66, k=1)
    fc[:, rows, cols] = x_f
    fc[:, cols, rows] = x_f
    return fc


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TCGNetDataset(ConnectomeDataset):
    def __init__(self, fc_matrices, graph_feats, y, group_mask, x_p):
        super().__init__(fc_matrices, graph_feats, y, group_mask)
        self.x_p = torch.as_tensor(x_p, dtype=torch.float32)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.prot_x = self.x_p[idx].unsqueeze(0)  # [1, 2920] → PyG batches to [B, 2920]
        return data


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)

def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    JOINT_PT   = Path('/shared/tcgnet_data/dataset_joint_fluid_iq.pt')
    FEAT_CACHE = Path('/shared/tcgnet_data/graph_feats_joint.npz')
    FMRI_CKPT  = Path('/shared/tcgnet_data/fmri_stage1_encoder.pt')
    PROT_CKPT  = Path('/shared/tcgnet_data/proteomics_stage1.pt')
    SAVE_DIR   = REPO_ROOT / 'fusion' / 'training' / 'checkpoints'
    BATCH_SIZE = 32

    # ------------------------------------------------------------------
    # Step 2 — Data setup
    # ------------------------------------------------------------------
    ds        = torch.load(JOINT_PT, weights_only=False)
    x_p       = ds['x_p'].numpy()
    x_f_raw   = ds['x_f'].numpy()
    y         = ds['y'].numpy()
    train_idx = ds['train_idx'].numpy().astype(int)
    val_idx   = ds['val_idx'].numpy().astype(int)
    test_idx  = ds['test_idx'].numpy().astype(int)

    fc_matrices = reconstruct_fc(x_f_raw)
    group_mask  = build_group_mask(fc_matrices, train_idx)
    graph_feats = compute_graph_features(fc_matrices, train_idx, group_mask, str(FEAT_CACHE))

    dataset = TCGNetDataset(fc_matrices, graph_feats, y, group_mask, x_p)
    dataset.apply_normalization(train_idx)

    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx.tolist()),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(Subset(dataset, test_idx.tolist()),  batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # Step 3 — Model assembly
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fmri_enc = ConnectomeEncoder().to(device)
    fmri_enc.load_state_dict(torch.load(FMRI_CKPT, map_location=device, weights_only=False))

    prot_enc = ProteomicsEncoder().to(device)
    prot_ckpt = torch.load(PROT_CKPT, map_location=device, weights_only=False)
    prot_enc.load_state_dict(prot_ckpt['encoder_state_dict'])

    # Phase 1: T=1.0 (soft gate — prevents extreme winner-take-all from noisy random aux heads)
    # Phase 2: T→0.1 (sharp gate — aux heads now calibrated, Shapley values meaningful)
    fusion = CooperativeShapleyFusion(dropout=0.4, temperature=1.0).to(device)
    head   = PredictionHead(dropout=0.4).to(device)

    # ------------------------------------------------------------------
    # Step 4 — Set baseline
    # ------------------------------------------------------------------
    fusion.set_baseline(float(dataset.y[train_idx].mean()))

    # ------------------------------------------------------------------
    # Step 5 — Freeze encoders (Phase 1: fusion + head only)
    # ------------------------------------------------------------------
    freeze(fmri_enc)
    freeze(prot_enc)

    # ------------------------------------------------------------------
    # Step 6 — Optimizer (two-phase AdamW)
    # ------------------------------------------------------------------
    criterion = Stage2Loss()

    # DEVIATION from §7.2: Phase-1 lr=1e-4 (spec says 5e-4).
    opt = torch.optim.AdamW(
        list(fusion.parameters()) + list(head.parameters()),
        lr=1e-4, weight_decay=1e-3,
    )
    phase = 1  # switches to 2 at epoch 11

    # ------------------------------------------------------------------
    # Step 7 — Training loop
    # ------------------------------------------------------------------
    MAX_EPOCHS = 200
    best_val_mae = float('inf')
    patience_counter = 0
    PATIENCE = 20
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SAVE_DIR / 'training_log.jsonl'

    for epoch in range(1, MAX_EPOCHS + 1):

        # Phase 2 switch at epoch 11
        if epoch == 11 and phase == 1:
            unfreeze(fmri_enc)
            unfreeze(prot_enc)
            opt = torch.optim.AdamW([
                {'params': list(fmri_enc.parameters()) + list(prot_enc.parameters()), 'lr': 1e-5},
                {'params': list(fusion.parameters())   + list(head.parameters()),     'lr': 1e-4},
            ], weight_decay=1e-3)
            phase = 2
            patience_counter = 0  # reset: encoders now trainable, give model fresh start
            fusion.temperature.fill_(0.1)  # sharpen gate: aux heads now calibrated
            print('Epoch 011 [phase] switched to Phase 2 — encoders unfrozen, T→0.1, patience reset')

        # --- train ---
        fmri_enc.train(); prot_enc.train(); fusion.train(); head.train()
        train_totals = {}
        train_mae_sum = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            z_f              = fmri_enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            z_p, s_p         = prot_enc(batch.prot_x)
            fusion_out       = fusion(z_f, z_p)
            y_hat            = head(fusion_out.z_fusion)
            alpha_f          = fmri_enc.last_attn[1]

            loss_dict        = criterion(batch.y, y_hat, fusion_out, s_p, alpha_f, fusion)
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(
                list(fusion.parameters()) + list(head.parameters()) +
                list(fmri_enc.parameters()) + list(prot_enc.parameters()),
                max_norm=1.0,
            )
            opt.step()
            opt.zero_grad()

            for k, v in loss_dict.items():
                train_totals[k] = train_totals.get(k, 0.0) + v.item()
            train_mae_sum += (y_hat.detach() - batch.y).abs().mean().item()

        n_train = len(train_loader)
        train_mae = train_mae_sum / n_train

        # --- validate ---
        fmri_enc.eval(); prot_enc.eval(); fusion.eval(); head.eval()
        val_mae_sum = 0.0
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                z_f        = fmri_enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                z_p, s_p   = prot_enc(batch.prot_x)
                fusion_out = fusion(z_f, z_p)
                y_hat      = head(fusion_out.z_fusion)
                alpha_f    = fmri_enc.last_attn[1]

                loss_dict  = criterion(batch.y, y_hat, fusion_out, s_p, alpha_f, fusion)
                val_loss_sum += loss_dict['total'].item()
                val_mae_sum  += (y_hat - batch.y).abs().mean().item()

        val_mae     = val_mae_sum  / len(val_loader)
        val_loss    = val_loss_sum / len(val_loader)
        val_mae_iq  = val_mae * dataset.y_std
        print(f'Epoch {epoch:03d} [p{phase}]  train_mae {train_mae:.4f} | val_mae {val_mae:.4f} ({val_mae_iq:.3f} IQ) | loss {val_loss:.4f}')

        # --- log to file ---
        log_entry = {
            'epoch':        epoch,
            'phase':        phase,
            'train_mae_z':  round(train_mae, 6),
            'train_mae_iq': round(train_mae * dataset.y_std, 6),
            **{f'train_{k}': round(v / len(train_loader), 6) for k, v in train_totals.items()},
            'val_total':    round(val_loss, 6),
            'val_mae_z':    round(val_mae, 6),
            'val_mae_iq':   round(val_mae_iq, 6),
            'best_val_mae': round(best_val_mae, 6),
            'patience':     patience_counter,
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # --- checkpoint + early stopping ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                'epoch':     epoch,
                'fmri_enc':  fmri_enc.state_dict(),
                'prot_enc':  prot_enc.state_dict(),
                'fusion':    fusion.state_dict(),
                'head':      head.state_dict(),
                'optimizer': opt.state_dict(),
                'val_mae':   best_val_mae,
            }, SAVE_DIR / 'best_model.pt')
            print(f'  → checkpoint saved (best val_mae {best_val_mae:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch} (patience={PATIENCE})')
                break

    print(f'Training complete. Best val MAE: {best_val_mae:.4f}')


if __name__ == '__main__':
    main()
