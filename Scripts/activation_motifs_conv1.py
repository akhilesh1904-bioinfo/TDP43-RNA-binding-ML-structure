import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import heapq

MODEL_PATH = "cnn_best.pt"
DATA_CSV   = "tdp43_sequences_labeled.csv"
OUTDIR     = Path("interpret_cnn")
OUTDIR.mkdir(exist_ok=True)
(OUTDIR / "arrays").mkdir(exist_ok=True)
(OUTDIR / "figures").mkdir(exist_ok=True)
(OUTDIR / "top_windows").mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model definition (matches your saved state_dict) ----
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

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.gpool(x).flatten(1)
        return self.classifier(x)

# Load weights
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model = CNNModel()

# match classifier dims exactly
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

# We interpret conv1 filters using activations after bn1+ReLU (what the network "uses")
K = model.conv1.kernel_size[0]  # 7
FILT = model.conv1.out_channels  # 64
print("Loaded model. conv1 kernel:", K, "filters:", FILT, "device:", device)

# ---- Data ----
df = pd.read_csv(DATA_CSV)
seqs = df["sequence"].astype(str).tolist()
labels = df["label"].astype(int).to_numpy()

BASES = "ACGT"
b2i = {b:i for i,b in enumerate(BASES)}

def onehot(seq: str) -> np.ndarray:
    """Return (4, L) float32 one-hot. Unknown/N -> all zeros."""
    L = len(seq)
    X = np.zeros((4, L), dtype=np.float32)
    for j,ch in enumerate(seq):
        idx = b2i.get(ch, None)
        if idx is not None:
            X[idx, j] = 1.0
    return X

# ---- Top-activating windows per filter (controlled scope) ----
TOP_N = 2000          # good depth, still controlled
BATCH = 128           # adjust if GPU RAM is low
MIN_ACT = 0.0         # keep 0 initially; you can raise later (e.g., 0.2)

# For each filter, store a min-heap of (activation, global_index, pos, label)
heaps = [[] for _ in range(FILT)]

def push_top(f, item):
    # item is (act, global_i, pos, label)
    if item[0] <= MIN_ACT:
        return
    h = heaps[f]
    if len(h) < TOP_N:
        heapq.heappush(h, item)
    else:
        if item[0] > h[0][0]:
            heapq.heapreplace(h, item)

# Iterate batches with dynamic padding
N = len(seqs)
for start in range(0, N, BATCH):
    end = min(N, start + BATCH)
    batch_seqs = seqs[start:end]
    batch_y = labels[start:end]
    lens = np.array([len(s) for s in batch_seqs], dtype=int)
    Lmax = int(lens.max())
    if Lmax < K:
        continue

    Xb = np.zeros((end-start, 4, Lmax), dtype=np.float32)
    for i,s in enumerate(batch_seqs):
        oh = onehot(s)
        Xb[i, :, :oh.shape[1]] = oh

    X_t = torch.tensor(Xb, device=device)
    with torch.no_grad():
        # conv1 -> bn1 -> relu
        a = model.relu(model.bn1(model.conv1(X_t)))  # (B, FILT, Lmax-K+1)
        a = a.detach().cpu().numpy()

    out_len = a.shape[2]  # Lmax-K+1
    # For each sequence in batch, only positions up to (len(seq)-K) are valid
    valid_maxpos = lens - K  # max start index for a K-window
    for bi in range(end-start):
        vmax = valid_maxpos[bi]
        if vmax < 0:
            continue
        # valid positions are 0..vmax inclusive, which corresponds to a[:, :, :vmax+1]
        for f in range(FILT):
            # take top few positions in this sequence for this filter (controls runtime)
            vec = a[bi, f, :vmax+1]
            if vec.size == 0:
                continue
            # grab top 3 per sequence per filter
            k_local = 3 if vec.size >= 3 else vec.size
            idxs = np.argpartition(vec, -k_local)[-k_local:]
            for p in idxs:
                push_top(f, (float(vec[p]), start + bi, int(p), int(batch_y[bi])))

    if (start // BATCH) % 20 == 0:
        print(f"Processed {end}/{N}")

print("Finished scanning activations.")

# Convert heaps to sorted lists (descending)
top_hits = []
for f in range(FILT):
    hits = sorted(heaps[f], key=lambda x: x[0], reverse=True)
    for rank,(act, gi, pos, lab) in enumerate(hits, start=1):
        top_hits.append({"filter": f, "rank": rank, "activation": act, "index": gi, "pos": pos, "label": lab})

top_hits_df = pd.DataFrame(top_hits)
top_hits_df.to_csv(OUTDIR / "top_windows" / "conv1_top_hits.csv", index=False)
print("Saved top hits:", OUTDIR / "top_windows" / "conv1_top_hits.csv")

# Build PWMs from the extracted windows
pwms_all = np.zeros((FILT, 4, K), dtype=np.float32)
pwms_pos = np.zeros((FILT, 4, K), dtype=np.float32)
pwms_neg = np.zeros((FILT, 4, K), dtype=np.float32)
counts_all = np.zeros(FILT, dtype=int)
counts_pos = np.zeros(FILT, dtype=int)
counts_neg = np.zeros(FILT, dtype=int)
max_act = np.zeros(FILT, dtype=np.float32)

for f in range(FILT):
    hits = sorted(heaps[f], key=lambda x: x[0], reverse=True)
    if not hits:
        continue
    max_act[f] = hits[0][0]
    for (act, gi, pos, lab) in hits:
        s = seqs[gi]
        if pos + K > len(s):
            continue
        w = onehot(s[pos:pos+K])  # (4,K)
        pwms_all[f] += w
        counts_all[f] += 1
        if lab == 1:
            pwms_pos[f] += w
            counts_pos[f] += 1
        else:
            pwms_neg[f] += w
            counts_neg[f] += 1

# Normalize to probabilities per position
def norm_pwm(pwm, cnt):
    if cnt == 0:
        return pwm
    pwm = pwm / float(cnt)
    # ensure columns sum to 1 if there were Ns (all-zero cols)
    colsum = pwm.sum(axis=0, keepdims=True)
    colsum[colsum == 0] = 1.0
    return pwm / colsum

pwms_all_n = np.zeros_like(pwms_all)
pwms_pos_n = np.zeros_like(pwms_pos)
pwms_neg_n = np.zeros_like(pwms_neg)
for f in range(FILT):
    pwms_all_n[f] = norm_pwm(pwms_all[f], counts_all[f])
    pwms_pos_n[f] = norm_pwm(pwms_pos[f], counts_pos[f])
    pwms_neg_n[f] = norm_pwm(pwms_neg[f], counts_neg[f])

np.save(OUTDIR / "arrays" / "conv1_pwms_all.npy", pwms_all_n)
np.save(OUTDIR / "arrays" / "conv1_pwms_pos.npy", pwms_pos_n)
np.save(OUTDIR / "arrays" / "conv1_pwms_neg.npy", pwms_neg_n)
np.save(OUTDIR / "arrays" / "conv1_counts_all.npy", counts_all)
np.save(OUTDIR / "arrays" / "conv1_max_activation.npy", max_act)

# Consensus strings
bases = np.array(list("ACGT"))
def consensus(pwm_4k):
    return "".join(bases[np.argmax(pwm_4k[:, j])] for j in range(pwm_4k.shape[1]))

summary = []
for f in range(FILT):
    summary.append({
        "filter": f,
        "max_activation": float(max_act[f]),
        "n_windows": int(counts_all[f]),
        "consensus_all": consensus(pwms_all_n[f]) if counts_all[f] else "",
        "consensus_pos": consensus(pwms_pos_n[f]) if counts_pos[f] else "",
        "consensus_neg": consensus(pwms_neg_n[f]) if counts_neg[f] else "",
        "pos_fraction_in_top": (counts_pos[f] / counts_all[f]) if counts_all[f] else np.nan
    })

summary_df = pd.DataFrame(summary).sort_values("max_activation", ascending=False)
summary_df.to_csv(OUTDIR / "filter_motifs_summary_conv1.csv", index=False)
print("Saved:", OUTDIR / "filter_motifs_summary_conv1.csv")

print(summary_df.head(12).to_string(index=False))
print("DONE")
