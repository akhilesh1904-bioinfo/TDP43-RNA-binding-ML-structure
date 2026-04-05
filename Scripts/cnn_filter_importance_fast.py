import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

MODEL_PATH = "cnn_best.pt"
DATA_CSV   = "tdp43_sequences_labeled.csv"
OUTDIR     = Path("interpret_cnn")
OUTDIR.mkdir(exist_ok=True)

device = "cpu"  # keep CPU unless you know CUDA works

# ---------- AUC (fast, no sklearn) ----------
def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan

    # rank scores; handle ties by average rank
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_score), dtype=np.float64) + 1.0

    # tie correction: average ranks for equal scores
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i]] + ranks[order[j-1]]) / 2.0
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

# ---------- Model definition (matches your state_dict keys) ----------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7)
        self.bn2   = nn.BatchNorm1d(128)

        self.relu  = nn.ReLU()
        self.gpool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward_from_a1(self, a1):
        # a1 is output of relu(bn1(conv1(x))) : (B,64,L1)
        x = self.relu(self.bn2(self.conv2(a1)))
        x = self.gpool(x).flatten(1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        a1 = self.relu(self.bn1(self.conv1(x)))
        return self.forward_from_a1(a1)

# load weights
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model = CNNModel()

# rebuild classifier dims exactly
w0 = state_dict["classifier.0.weight"]
w3 = state_dict["classifier.3.weight"]
model.classifier = nn.Sequential(
    nn.Linear(w0.shape[1], w0.shape[0]),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(w3.shape[1], w3.shape[0]),
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

K = model.conv1.kernel_size[0]
FILT = model.conv1.out_channels
print("Loaded model. K =", K, "filters =", FILT, "device =", device)

# ---------- Load data ----------
df = pd.read_csv(DATA_CSV)
seqs = df["sequence"].astype(str).tolist()
y = df["label"].astype(int).to_numpy()
N = len(seqs)

BASES = "ACGT"
b2i = {b:i for i,b in enumerate(BASES)}

def onehot(seq: str) -> np.ndarray:
    L = len(seq)
    X = np.zeros((4, L), dtype=np.float32)
    for j,ch in enumerate(seq):
        idx = b2i.get(ch, None)
        if idx is not None:
            X[idx, j] = 1.0
    return X

# ---------- Batched prediction with dynamic padding ----------
BATCH = 128

# store baseline predictions for full dataset (needed for baseline AUC)
baseline_scores = np.zeros(N, dtype=np.float32)

print("Computing baseline scores (batched)...")
with torch.no_grad():
    for start in range(0, N, BATCH):
        end = min(N, start + BATCH)
        batch_seqs = seqs[start:end]
        lens = np.array([len(s) for s in batch_seqs], dtype=int)
        Lmax = int(lens.max())
        if Lmax < K:
            # too short for conv kernel; baseline score stays 0.5-ish
            baseline_scores[start:end] = 0.5
            continue

        Xb = np.zeros((end-start, 4, Lmax), dtype=np.float32)
        for i,s in enumerate(batch_seqs):
            oh = onehot(s)
            Xb[i, :, :oh.shape[1]] = oh

        X_t = torch.tensor(Xb, device=device)
        logits = model(X_t).squeeze(1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        baseline_scores[start:end] = probs

        if (start // BATCH) % 50 == 0:
            print(f"  {end}/{N}")

baseline_auc = roc_auc(y, baseline_scores)
baseline_acc = float(((baseline_scores >= 0.5).astype(int) == y).mean())
print("Baseline AUC:", baseline_auc)
print("Baseline Acc:", baseline_acc)

# ---------- Filter ablation importance (FAST) ----------
# We compute AUC for each filter with that filter zeroed at a1 stage.
importance_rows = []

print("Computing filter importance (ablation on a1 channel, batched)...")
for f in range(FILT):
    ablated_scores = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, BATCH):
            end = min(N, start + BATCH)
            batch_seqs = seqs[start:end]
            lens = np.array([len(s) for s in batch_seqs], dtype=int)
            Lmax = int(lens.max())
            if Lmax < K:
                ablated_scores[start:end] = 0.5
                continue

            Xb = np.zeros((end-start, 4, Lmax), dtype=np.float32)
            for i,s in enumerate(batch_seqs):
                oh = onehot(s)
                Xb[i, :, :oh.shape[1]] = oh

            X_t = torch.tensor(Xb, device=device)

            # compute a1 once
            a1 = model.relu(model.bn1(model.conv1(X_t)))  # (B,64,L1)

            # ablate filter f
            a1[:, f, :] = 0.0

            # run remainder of network
            logits = model.forward_from_a1(a1).squeeze(1).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            ablated_scores[start:end] = probs

    auc_f = roc_auc(y, ablated_scores)
    acc_f = float(((ablated_scores >= 0.5).astype(int) == y).mean())

    importance_rows.append({
        "filter": f,
        "baseline_auc": baseline_auc,
        "ablated_auc": auc_f,
        "auc_drop": float(baseline_auc - auc_f),

        "baseline_acc": baseline_acc,
        "ablated_acc": acc_f,
        "acc_drop": float(baseline_acc - acc_f),
    })

    if (f+1) % 8 == 0:
        best_so_far = sorted(importance_rows, key=lambda r: r["auc_drop"], reverse=True)[:3]
        print(f"  done {f+1}/{FILT} | top auc_drop so far:",
              [(r["filter"], round(r["auc_drop"], 4)) for r in best_so_far])

imp_df = pd.DataFrame(importance_rows).sort_values("auc_drop", ascending=False)
out_csv = OUTDIR / "filter_importance_fast.csv"
imp_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("\nTop 12 filters by AUC drop:")
print(imp_df.head(12).to_string(index=False))
print("\nDONE")
