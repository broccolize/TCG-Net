import argparse
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
from proteomics.config import SHARED_DATA_DIR
from proteomics.utils.seed import make_split_generator, set_global_seeds
from proteomics.utils.logging_utils import make_run_id
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


def parse_args():
    p = argparse.ArgumentParser(description='TCG-Net Stage 2 (fluid IQ).')
    p.add_argument('--seed', type=int, default=42,
                   help='Global seed. Stage 2 was previously unseeded, so no run was reproducible.')
    p.add_argument('--weight-decay', type=float, default=1e-3, dest='weight_decay',
                   help='AdamW weight decay for both phases. Phase 2 overfits within ~3 epochs '
                        'of unfreezing at the 1e-3 default; sweep upward.')
    p.add_argument('--lr-head', type=float, default=1e-4, dest='lr_head',
                   help='LR for the fusion module and prediction head.')
    p.add_argument('--lr-encoder', type=float, default=1e-5, dest='lr_encoder',
                   help='LR for the encoders once unfrozen in Phase 2.')
    p.add_argument('--init-heads', action='store_true', dest='init_heads',
                   help='Warm-start the fusion H_p from the Stage-1 proteomics aux '
                        'head instead of random init. Stage 1 fits that head on the '
                        'full cohort; refitting it on 3.5k joint-train subjects '
                        'discards the pretraining. Off by default so earlier runs '
                        'stay reproducible.')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--phase2-epoch', type=int, default=11, dest='phase2_epoch',
                   help='Epoch at which encoders unfreeze and the gate sharpens to T=0.1.')
    p.add_argument('--no-test-eval', action='store_true', dest='no_test_eval',
                   help='Skip the held-out test evaluation after training.')
    return p.parse_args()


def fit_affine(p, t):
    """
    Least-squares scale and offset mapping a prediction onto its target.

    Returns (a, b) minimising ||t - (a*p + b)||^2. Fit this on the VALIDATION
    fold and apply it to test — fitting on test would just report r^2, which is
    a ceiling and not a number anyone can publish.
    """
    p = p.double(); t = t.double()
    pm, tm = p.mean(), t.mean()
    var = ((p - pm) ** 2).sum()
    if var <= 0:
        return 0.0, tm.item()          # collapsed predictor: best affine map is the mean
    a = (((p - pm) * (t - tm)).sum() / var).item()
    return a, (tm - a * pm).item()


def r2_of(p, t):
    """R^2 with SST against the evaluation set's own mean, as evaluate() uses."""
    ss_tot = ((t - t.mean()) ** 2).sum()
    if ss_tot <= 0:
        return float('nan')
    return (1.0 - ((t - p) ** 2).sum() / ss_tot).item()


def evaluate(loader, fmri_enc, prot_enc, fusion, head, criterion, device,
             return_preds=False):
    """
    Pooled evaluation over a loader.

    All predictions are concatenated before any metric is computed, so R^2 and
    Pearson r are not batch-size biased. The MAE this replaces was an average of
    per-batch means, which is slightly wrong whenever the last batch is short.

    Returns mae / loss / r2 / pearson_r on the z-scored target scale.
    """
    fmri_enc.eval(); prot_enc.eval(); fusion.eval(); head.eval()
    # The fused output plus the three internal branch predictions. Scoring the
    # branches is the only way to tell whether the fusion module adds anything
    # over its own inputs, or destroys signal that was already there.
    cols = {k: [] for k in ('y_hat', 'y_hat_f', 'y_hat_p', 'y_hat_fp')}
    targets = []
    loss_sum, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z_f        = fmri_enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            z_p, s_p   = prot_enc(batch.prot_x)
            fusion_out = fusion(z_f, z_p)
            y_hat      = head(fusion_out.z_fusion)
            alpha_f    = fmri_enc.last_attn[1]

            loss_sum += criterion(batch.y, y_hat, fusion_out, s_p, alpha_f, fusion)['total'].item()
            n_batches += 1
            cols['y_hat'].append(y_hat.detach().cpu())
            cols['y_hat_f'].append(fusion_out.y_hat_f.detach().cpu())
            cols['y_hat_p'].append(fusion_out.y_hat_p.detach().cpu())
            cols['y_hat_fp'].append(fusion_out.y_hat_fp.detach().cpu())
            targets.append(batch.y.detach().cpu())

    t = torch.cat(targets).flatten()
    ss_tot = ((t - t.mean()) ** 2).sum()

    def _score(p):
        r2 = (1.0 - ((t - p) ** 2).sum() / ss_tot).item() if ss_tot > 0 else float('nan')
        # Guard the degenerate case: a collapsed model predicts a constant, and
        # torch.corrcoef then divides by a zero standard deviation, returning nan.
        # Stage 1 hit exactly this at weight_decay=3e0.
        r = (torch.corrcoef(torch.stack([p, t]))[0, 1].item()
             if p.std() > 0 and t.std() > 0 else float('nan'))
        return r2, r, (p - t).abs().mean().item()

    p_main = torch.cat(cols['y_hat']).flatten()
    r2, pearson, mae = _score(p_main)

    out = {'mae': mae, 'loss': loss_sum / max(n_batches, 1), 'r2': r2, 'pearson_r': pearson}
    for name in ('y_hat_f', 'y_hat_p', 'y_hat_fp'):
        b_r2, b_r, b_mae = _score(torch.cat(cols[name]).flatten())
        out[f'r2_{name}'] = b_r2
        out[f'pearson_{name}'] = b_r
        out[f'mae_{name}'] = b_mae
    # Std of the fused prediction against the target's. A value far below 1.0
    # means the model has shrunk toward the mean rather than learned to rank.
    out['pred_std'] = p_main.std().item()
    out['target_std'] = t.std().item()
    if return_preds:
        # Raw predictions, for fitting/applying the affine calibration across
        # folds. Underscore-prefixed so they never reach the JSONL log.
        out['_preds'] = {k: torch.cat(v).flatten() for k, v in cols.items()}
        out['_targets'] = t
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Stage 2 was previously unseeded: weight init, dropout masks and shuffling
    # all differed every run, so no reported number could be reproduced.
    set_global_seeds(args.seed)
    run_id = make_run_id()
    print(f'[run] run_id={run_id} seed={args.seed} weight_decay={args.weight_decay} '
          f'lr_head={args.lr_head} lr_encoder={args.lr_encoder}')

    TARGET     = 'fluid_iq'
    JOINT_PT   = SHARED_DATA_DIR / 'dataset_joint_fluid_iq.pt'
    FEAT_CACHE = SHARED_DATA_DIR / 'graph_feats_joint.npz'
    FMRI_CKPT  = SHARED_DATA_DIR / 'fmri_stage1_encoder.pt'
    # Target-specific: both trainers previously pointed at one shared
    # proteomics_stage1.pt, so whichever target it was trained on, the other
    # run silently warm-started from the wrong encoder.
    PROT_CKPT  = SHARED_DATA_DIR / 'proteomics_stage1_fluid_iq.pt'
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

    # drop_last: ProteomicsEncoder normalises over the batch dimension, so a
    # trailing batch of 1 raises inside BatchNorm1d in train mode.
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True,
                              generator=make_split_generator(args.seed))
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
    ckpt_target = prot_ckpt.get('config', {}).get('target')
    if ckpt_target != TARGET:
        raise ValueError(
            f"{PROT_CKPT.name} was trained on target={ckpt_target!r}, but this is "
            f"the {TARGET!r} run. Loading it would warm-start the proteomics branch "
            f"from a model fit to a different target. Point PROT_CKPT at the "
            f"{TARGET!r} Stage-1 checkpoint."
        )
    print(f'[ckpt] proteomics encoder: {PROT_CKPT.name} '
          f"(target={ckpt_target}, epoch={prot_ckpt.get('epoch')}, "
          f"val_r2={prot_ckpt.get('val_metrics', {}).get('val_r2')})")
    prot_enc.load_state_dict(prot_ckpt['encoder_state_dict'])

    # Phase 1: T=1.0 (soft gate — prevents extreme winner-take-all from noisy random aux heads)
    # Phase 2: T→0.1 (sharp gate — aux heads now calibrated, Shapley values meaningful)
    fusion = CooperativeShapleyFusion(dropout=0.4, temperature=1.0).to(device)
    head   = PredictionHead(dropout=0.4).to(device)

    # ------------------------------------------------------------------
    # Optional: warm-start H_p from the Stage-1 auxiliary head.
    #
    # Stage 1 fits its aux head on the full single-modality cohort (11,528
    # subjects for fluid IQ); Stage 2 throws it away and refits H_p from random
    # init on 3,471 joint-train subjects. Measured 2026-08-01: ridge pretrained
    # on the same large cohort and then merely affine-adapted on joint train
    # scores 0.0363 on the joint test, against 0.0294 for the calibrated
    # Stage-2 proteomics branch — i.e. discarding the pretrained head costs more
    # than the encoder gains. ProteomicsAuxHead is nn.Linear(128, 1), the same
    # shape as H_p, so the weights transfer directly.
    #
    # Off by default so every run before this date stays reproducible.
    # ------------------------------------------------------------------
    if args.init_heads:
        aux_sd = prot_ckpt.get('aux_head_state_dict')
        if aux_sd is None:
            print('[init-heads] WARNING: no aux_head_state_dict in the proteomics '
                  'checkpoint — H_p left at random init.')
        else:
            fusion.H_p.load_state_dict(
                {'weight': aux_sd['fc.weight'], 'bias': aux_sd['fc.bias']}
            )
            print('[init-heads] H_p warm-started from the Stage-1 proteomics aux head.')
        # The fMRI checkpoint is a bare encoder state dict, so there is no
        # pretrained H_f to transfer. Saving it is on the fMRI side (item 15).
        print('[init-heads] H_f left at random init — the fMRI checkpoint stores '
              'no aux head.')

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
        lr=args.lr_head, weight_decay=args.weight_decay,
    )
    phase = 1  # switches to 2 at args.phase2_epoch

    # ------------------------------------------------------------------
    # Step 7 — Training loop
    # ------------------------------------------------------------------
    MAX_EPOCHS = args.epochs
    best_val_mae = float('inf')
    best_epoch = -1
    patience_counter = 0
    PATIENCE = args.patience
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SAVE_DIR / 'training_log.jsonl'

    for epoch in range(1, MAX_EPOCHS + 1):

        # Phase 2 switch at epoch 11
        if epoch == args.phase2_epoch and phase == 1:
            unfreeze(fmri_enc)
            unfreeze(prot_enc)
            opt = torch.optim.AdamW([
                {'params': list(fmri_enc.parameters()) + list(prot_enc.parameters()),
                 'lr': args.lr_encoder},
                {'params': list(fusion.parameters()) + list(head.parameters()),
                 'lr': args.lr_head},
            ], weight_decay=args.weight_decay)
            phase = 2
            patience_counter = 0  # reset: encoders now trainable, give model fresh start
            fusion.temperature.fill_(0.1)  # sharpen gate: aux heads now calibrated
            print('Epoch 011 [phase] switched to Phase 2 — encoders unfrozen, T→0.1, patience reset')

        # --- train ---
        # Phase 1 keeps the encoders in eval mode. freeze() only clears
        # requires_grad, which does NOT stop BatchNorm running-statistic updates
        # (those are buffers, not parameters) and does not disable dropout.
        # Calling .train() on them overwrote the pretrained statistics with
        # joint-cohort ones from epoch 1, so the "frozen" encoders were never
        # frozen and Phase-1 validation measured a moving target.
        if phase == 1:
            fmri_enc.eval(); prot_enc.eval()
        else:
            fmri_enc.train(); prot_enc.train()
        fusion.train(); head.train()
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
        val         = evaluate(val_loader, fmri_enc, prot_enc, fusion, head, criterion, device)
        val_mae     = val['mae']
        val_loss    = val['loss']
        val_mae_iq  = val_mae * dataset.y_std
        print(f'Epoch {epoch:03d} [p{phase}]  train_mae {train_mae:.4f} | '
              f'val_mae {val_mae:.4f} ({val_mae_iq:.3f} IQ) | '
              f'val_r2 {val["r2"]:.4f} | loss {val_loss:.4f}')

        # --- log to file ---
        log_entry = {
            'run_id':       run_id,   # runs are appended to one file; without this
            'seed':         args.seed,  # they can only be separated by position
            'weight_decay': args.weight_decay,
            'epoch':        epoch,
            'phase':        phase,
            'train_mae_z':  round(train_mae, 6),
            'train_mae_iq': round(train_mae * dataset.y_std, 6),
            **{f'train_{k}': round(v / len(train_loader), 6) for k, v in train_totals.items()},
            'val_total':    round(val_loss, 6),
            'val_mae_z':    round(val_mae, 6),
            'val_mae_iq':   round(val_mae_iq, 6),
            'val_r2':       round(val['r2'], 6),
            'val_pearson_r': round(val['pearson_r'], 6),
            'best_val_mae': round(best_val_mae, 6),
            'patience':     patience_counter,
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # --- checkpoint + early stopping ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'run_id':    run_id,
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

    print(f'Training complete. Best val MAE: {best_val_mae:.4f} (epoch {best_epoch})')

    # ------------------------------------------------------------------
    # Step 8 — Held-out test evaluation
    #
    # Previously test_loader was constructed and never used, so every reported
    # Stage-2 number was a validation minimum that had also selected the
    # checkpoint and driven early stopping — optimistically biased by selection.
    # ------------------------------------------------------------------
    if args.no_test_eval:
        print('[test] skipped (--no-test-eval)')
        return

    ckpt_path = SAVE_DIR / 'best_model.pt'
    if best_epoch == -1 or not ckpt_path.exists():
        print('[test] no checkpoint was written — nothing to evaluate.')
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fmri_enc.load_state_dict(ckpt['fmri_enc'])
    prot_enc.load_state_dict(ckpt['prot_enc'])
    fusion.load_state_dict(ckpt['fusion'])
    head.load_state_dict(ckpt['head'])

    test = evaluate(test_loader, fmri_enc, prot_enc, fusion, head, criterion, device,
                    return_preds=True)

    # ------------------------------------------------------------------
    # Affine recalibration, fit on val and applied to test.
    #
    # Measured over 3 converged seeds (2026-08-01): the fused output ranks
    # better than the fMRI branch on every seed (r +0.019 mean) while being
    # more over-shrunk on every seed, so the two cancel and R^2 shows nothing.
    # r^2 - R^2 quantifies what the miscalibration costs, but r^2 is fitted on
    # the test set and is only a ceiling. This recovers the honest version:
    # the scale comes from val, the score from test.
    #
    # Caveat to carry into the write-up: val also drove early stopping and
    # checkpoint selection, so these numbers are not perfectly clean either.
    # They are, however, the standard construction and vastly better than r^2.
    # ------------------------------------------------------------------
    val_c = evaluate(val_loader, fmri_enc, prot_enc, fusion, head, criterion, device,
                     return_preds=True)
    cal, cal_scale = {}, {}
    for key, label in (('y_hat', ''), ('y_hat_f', '_fmri_only'),
                       ('y_hat_p', '_prot_only'), ('y_hat_fp', '_joint_head')):
        a, b = fit_affine(val_c['_preds'][key], val_c['_targets'])
        cal[label] = r2_of(a * test['_preds'][key].double() + b,
                           test['_targets'].double())
        cal_scale[label] = a
    print(f"[test] n={len(test_idx)} epoch={ckpt['epoch']}  "
          f"mae {test['mae']:.4f} ({test['mae'] * dataset.y_std:.3f} IQ) | "
          f"r2 {test['r2']:.4f} | r {test['pearson_r']:.4f} | loss {test['loss']:.4f}")
    print(f"[test] branch r2 — fused {test['r2']:.4f} | "
          f"fMRI-only {test['r2_y_hat_f']:.4f} | "
          f"prot-only {test['r2_y_hat_p']:.4f} | "
          f"joint-head {test['r2_y_hat_fp']:.4f}")
    print(f"[test] branch r  — fused {test['pearson_r']:.4f} | "
          f"fMRI-only {test['pearson_y_hat_f']:.4f} | "
          f"prot-only {test['pearson_y_hat_p']:.4f} | "
          f"joint-head {test['pearson_y_hat_fp']:.4f}")
    print(f"[test] pred_std {test['pred_std']:.4f} vs target_std {test['target_std']:.4f} "
          f"(far below 1.0 = shrunk toward the mean)")
    print(f"[test] val-calibrated r2 — fused {cal['']:.4f} | "
          f"fMRI-only {cal['_fmri_only']:.4f} | "
          f"prot-only {cal['_prot_only']:.4f} | "
          f"joint-head {cal['_joint_head']:.4f}")
    print(f"[test] calibration scale (>1 = raw output was over-shrunk) — "
          f"fused {cal_scale['']:.3f} | fMRI-only {cal_scale['_fmri_only']:.3f} | "
          f"prot-only {cal_scale['_prot_only']:.3f} | "
          f"joint-head {cal_scale['_joint_head']:.3f}")

    summary = {
        'run_id':        run_id,
        'target':        TARGET,
        'seed':          args.seed,
        'weight_decay':  args.weight_decay,
        'lr_head':       args.lr_head,
        'lr_encoder':    args.lr_encoder,
        'best_epoch':    best_epoch,
        'n_train':       int(len(train_idx)),
        'n_val':         int(len(val_idx)),
        'n_test':        int(len(test_idx)),
        'val_mae_z':     round(best_val_mae, 6),
        'test_mae_z':    round(test['mae'], 6),
        'test_mae_iq':   round(test['mae'] * dataset.y_std, 6),
        'test_r2':       round(test['r2'], 6),
        'test_pearson_r': round(test['pearson_r'], 6),
        # Branch scores: does the fusion beat its own inputs, or destroy them?
        'test_r2_fmri_only':  round(test['r2_y_hat_f'], 6),
        'test_r2_prot_only':  round(test['r2_y_hat_p'], 6),
        'test_r2_joint_head': round(test['r2_y_hat_fp'], 6),
        # Pearson r alongside R^2: a shrunk branch can rank well (r > 0) while
        # scoring R^2 ~ 0, so R^2 alone reads as "no signal" when there is some.
        # The proteomics branch is exactly this case (R^2 -0.0007, r ~ 0.17).
        'test_pearson_fmri_only':  round(test['pearson_y_hat_f'], 6),
        'test_pearson_prot_only':  round(test['pearson_y_hat_p'], 6),
        'test_pearson_joint_head': round(test['pearson_y_hat_fp'], 6),
        'test_pred_std':      round(test['pred_std'], 6),
        'test_target_std':    round(test['target_std'], 6),
        # Affine rescale fit on val, applied to test. The fused branch ranks
        # better than fMRI-only but is more over-shrunk; raw R^2 hides that.
        'test_r2_cal':             round(cal[''], 6),
        'test_r2_cal_fmri_only':   round(cal['_fmri_only'], 6),
        'test_r2_cal_prot_only':   round(cal['_prot_only'], 6),
        'test_r2_cal_joint_head':  round(cal['_joint_head'], 6),
        'cal_scale':             round(cal_scale[''], 6),
        'cal_scale_fmri_only':   round(cal_scale['_fmri_only'], 6),
        'cal_scale_prot_only':   round(cal_scale['_prot_only'], 6),
        'cal_scale_joint_head':  round(cal_scale['_joint_head'], 6),
        'prot_ckpt':     str(PROT_CKPT),
        'fmri_ckpt':     str(FMRI_CKPT),
        'joint_dataset': str(JOINT_PT),
    }
    summary_path = SAVE_DIR / f'test_metrics_{TARGET}_{run_id}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[test] summary written: {summary_path}')


if __name__ == '__main__':
    main()
