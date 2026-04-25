"""
TabM trainer (Gorishniy et al. 2024) on the Path-B genome.

TabM is a parameter-efficient deep ensemble of MLPs — it trains k MLPs in
parallel sharing most weights, then averages their predictions. Reportedly
matches/beats tuned GBDT on small-n tabular (n<10k).

Usage:
    python3 scripts/train_tabm.py
    python3 scripts/train_tabm.py --epochs 100 --k 32 --d-block 512

Compares OOF metrics to the XGBoost baseline reported in qa_report.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.early_features import compute_early_features
from src.data.feature_engineering import TabularFeatureEngineer
from src.data.loader import DataLoader
from src.data.rubric_features import align_rubric
from src.embeddings.clip_encoder import EmbeddingCache
from tabm import TabM

CONFIG = "config.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATUS_LABELS = ["top_performer", "stable", "fatigued", "underperformer"]


def build_genome():
    """Returns X, y_perf, y_status, groups, feature_names."""
    loader = DataLoader(CONFIG)
    df = loader.load_master_table()
    daily = loader.load_daily_stats()
    eng = TabularFeatureEngineer()
    X_tab, names = eng.fit_transform(df)
    y_perf = eng.get_perf_scores(df)
    y_status = eng.get_status_labels(df)
    groups = df["campaign_id"].values

    cids = df["creative_id"].astype(int).tolist()
    X_early, early_names = compute_early_features(daily, cids, window=7)
    X_rubric, rubric_names = align_rubric("outputs/rubric/rubric_scores.parquet", cids)

    cache = EmbeddingCache("outputs/embeddings/clip_embeddings.npz")
    emb, ids = cache.load()
    emb_dim = emb.shape[1]
    id2 = {c: emb[i] for i, c in enumerate(ids)}
    X_clip = np.stack([id2.get(int(c), np.zeros(emb_dim)) for c in df["creative_id"]])

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    n_pca = int(cfg.get("embeddings", {}).get("pca_components", 32))
    X_clip_r = PCA(n_components=n_pca).fit_transform(X_clip)

    X = np.concatenate([X_tab, X_early, X_rubric, X_clip_r], axis=1).astype(np.float32)
    feature_names = (names + early_names + rubric_names
                     + [f"clip_pc{i}" for i in range(n_pca)])
    return X, y_perf, y_status.astype(np.int64), groups, feature_names


def fit_tabm(
    X_tr, y_tr, X_va, y_va,
    *,
    task: str,           # "classification" or "regression"
    n_classes: int = 1,  # ignored for regression
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    k: int = 32,
    n_blocks: int = 3,
    d_block: int = 512,
    dropout: float = 0.1,
    patience: int = 15,
    sample_weight=None,
    verbose: bool = False,
):
    """Train one TabM model and return (model, scaler) for the val set
    plus the OOF prediction array for X_va."""
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)

    n_in = X_tr_s.shape[1]
    d_out = n_classes if task == "classification" else 1
    model = TabM(
        n_num_features=n_in,
        cat_cardinalities=None,
        d_out=d_out,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        k=k,
        arch_type="tabm",
        start_scaling_init="random-signs",
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if task == "classification":
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")

    X_tr_t = torch.tensor(X_tr_s, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va_s, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)
    if sample_weight is not None:
        sw_t = torch.tensor(sample_weight.astype(np.float32), device=DEVICE)
    else:
        sw_t = None

    n_train = len(y_tr)
    rng = np.random.default_rng(42)
    best_val = float("inf") if task == "regression" else -float("inf")
    best_state = None
    epochs_no_imp = 0

    for ep in range(epochs):
        model.train()
        idx = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            sel = idx[start: start + batch_size]
            xb = X_tr_t[sel]
            yb = y_tr_t[sel]
            wb = sw_t[sel] if sw_t is not None else None

            logits = model(xb)  # shape (B, k, d_out)
            if task == "classification":
                # logits: (B, k, C) → flatten to (B*k, C); targets: (B,) → (B*k,)
                logits_flat = logits.reshape(-1, d_out)
                targets_flat = yb.repeat_interleave(k)
                losses = loss_fn(logits_flat, targets_flat).reshape(len(yb), k)
            else:
                losses = loss_fn(logits.squeeze(-1), yb.unsqueeze(1).expand(-1, k).float())
            if wb is not None:
                losses = losses * wb.unsqueeze(1)
            loss = losses.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_va = model(X_va_t)  # (B, k, d_out)
            if task == "classification":
                # Average probs across the k ensemble members
                probs_va = torch.softmax(logits_va, dim=-1).mean(dim=1)  # (B, d_out)
                pred_va = probs_va.argmax(dim=1)
                cur = -loss_fn(probs_va.log(), y_va_t).mean().item()  # higher better
            else:
                pred_va = logits_va.mean(dim=1).squeeze(-1)
                cur = -mean_absolute_error(y_va, pred_va.cpu().numpy())  # higher better

        improved = cur > best_val if task == "classification" else cur > best_val
        if improved:
            best_val = cur
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                if verbose:
                    print(f"  early stop at ep {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final prediction
    model.eval()
    with torch.no_grad():
        logits_va = model(X_va_t)
        if task == "classification":
            probs_va = torch.softmax(logits_va, dim=-1).mean(dim=1).cpu().numpy()
            return probs_va  # (B, n_classes)
        else:
            return logits_va.mean(dim=1).squeeze(-1).cpu().numpy()


def run(args):
    X, y_perf, y_status, groups, feat_names = build_genome()
    print(f"Genome: {X.shape}  (status counts: {np.bincount(y_status)})")

    n_total = len(X)
    n_classes = 4

    # ---- 5-fold OOF for classifier ----
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((n_total, n_classes), dtype=np.float32)
    oof_pred = np.full(n_total, -1, dtype=int)

    # Just plain balanced — the 1.7× boost was tuned for XGBoost's tree splits
    # and over-corrects on a soft DL classifier.
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y_status).astype(np.float32)

    t0 = time.time()
    for fold, (tr, va) in enumerate(sgkf.split(X, y_status, groups)):
        probs_va = fit_tabm(
            X[tr], y_status[tr], X[va], y_status[va],
            task="classification", n_classes=n_classes,
            epochs=args.epochs, k=args.k, d_block=args.d_block,
            sample_weight=sw[tr], verbose=False,
        )
        oof_probs[va] = probs_va
        oof_pred[va] = probs_va.argmax(axis=1)
        print(f"  fold {fold}: val acc = {(oof_pred[va] == y_status[va]).mean():.3f}")
    print(f"  classifier OOF time: {time.time()-t0:.1f}s")

    print("\nStatus report (TabM, OOF, StratifiedGroupKFold):")
    print(classification_report(y_status, oof_pred, target_names=STATUS_LABELS))

    # ---- 5-fold OOF for regressor ----
    print("\n=== Perf score regressor (TabM) ===")
    gkf = GroupKFold(n_splits=5)
    oof_perf = np.zeros(n_total, dtype=np.float32)
    t0 = time.time()
    for fold, (tr, va) in enumerate(gkf.split(X, y_perf, groups)):
        pred = fit_tabm(
            X[tr], y_perf[tr], X[va], y_perf[va],
            task="regression",
            epochs=args.epochs, k=args.k, d_block=args.d_block,
        )
        oof_perf[va] = pred
        print(f"  fold {fold}: val MAE = {mean_absolute_error(y_perf[va], pred):.4f}")
    print(f"  regressor OOF time: {time.time()-t0:.1f}s")
    print(f"\nCV Perf MAE (TabM, OOF, GroupKFold): {mean_absolute_error(y_perf, oof_perf):.4f}")

    # Save the OOF arrays for diff against XGBoost
    out = Path("outputs/models/tabm_oof.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        oof_status_pred=oof_pred,
        oof_status_probs=oof_probs,
        oof_perf=oof_perf,
        y_status=y_status,
        y_perf=y_perf,
    )
    print(f"\nSaved OOF predictions to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--d-block", type=int, default=512)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
