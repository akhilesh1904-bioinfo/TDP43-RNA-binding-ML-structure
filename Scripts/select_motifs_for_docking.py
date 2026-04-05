import pandas as pd
import numpy as np
from pathlib import Path

OUTDIR = Path("interpret_cnn")
OUTDIR.mkdir(exist_ok=True)

# Inputs you already have
motifs = pd.read_csv("interpret_cnn/filter_motifs_summary_conv1.csv")               # CNN motifs
imp    = pd.read_csv("interpret_cnn/filter_importance_fast.csv")                   # CNN importance
kmers  = pd.read_csv("top_kmers.csv")                                              # LR kmers (your file)

# --- robustly identify LR k-mer column names ---
# Common possibilities: kmer, motif, sequence; importance/coef/weight
kmer_col = None
for c in kmers.columns:
    if c.lower() in ["kmer", "motif", "sequence", "seq"]:
        kmer_col = c
        break
if kmer_col is None:
    # fallback: first object column
    obj_cols = [c for c in kmers.columns if kmers[c].dtype == object]
    kmer_col = obj_cols[0]

score_col = None
for c in kmers.columns:
    if any(x in c.lower() for x in ["importance", "coef", "weight", "score"]):
        score_col = c
        break

kmers_clean = kmers.copy()
kmers_clean[kmer_col] = kmers_clean[kmer_col].astype(str).str.upper()

# Take top LR kmers (keep it controlled)
TOP_LR = 50
if score_col:
    top_lr = kmers_clean.sort_values(score_col, ascending=False).head(TOP_LR)[kmer_col].tolist()
else:
    top_lr = kmers_clean.head(TOP_LR)[kmer_col].tolist()

# Merge CNN motif + importance
df = imp.merge(motifs, on="filter", how="inner")

# Score filters by both importance + enrichment
# (importance is auc_drop; enrichment is pos_fraction_in_top)
df["combo_score"] = df["auc_drop"].rank(pct=True) + df["pos_fraction_in_top"].rank(pct=True)

# Keep top candidate filters
cands = df.sort_values("combo_score", ascending=False).head(25).copy()

# Helper: match CNN 7-mer consensus against LR kmers (substring match)
def lr_support(consensus7):
    hits = []
    for k in top_lr:
        if k in consensus7 or consensus7 in k:
            hits.append(k)
    return hits

cands["lr_hits"] = cands["consensus_all"].apply(lr_support)
cands["n_lr_hits"] = cands["lr_hits"].apply(len)

# Prefer filters with LR support, then by combo_score
cands = cands.sort_values(["n_lr_hits", "combo_score"], ascending=False)

# Reduce redundancy: keep motifs that are not near-duplicates
selected = []
def is_redundant(m, chosen):
    # simple redundancy: shared 6/7 positions or one contained in other
    for c in chosen:
        same = sum(a==b for a,b in zip(m,c))
        if same >= 6:
            return True
        if m in c or c in m:
            return True
    return False

for _,r in cands.iterrows():
    m = r["consensus_all"]
    if not isinstance(m, str) or len(m) == 0:
        continue
    if not is_redundant(m, [x["dna_motif_7mer"] for x in selected]):
        selected.append({
            "source": "CNN",
            "filter": int(r["filter"]),
            "dna_motif_7mer": m,
            "rna_motif_7mer": m.replace("T","U"),
            "auc_drop": float(r["auc_drop"]),
            "pos_fraction_in_top": float(r["pos_fraction_in_top"]),
            "lr_supporting_kmers": ";".join(r["lr_hits"][:10])  # cap
        })
    if len(selected) >= 5:
        break

# If we somehow got <3, supplement with top LR kmers directly (nonredundant)
if len(selected) < 3:
    chosen = [x["dna_motif_7mer"] for x in selected]
    for k in top_lr:
        k = k.replace("U","T")  # if any RNA-like kmers
        if len(k) < 6:
            continue
        k7 = k[:7]
        if not is_redundant(k7, chosen):
            selected.append({
                "source": "LR",
                "filter": -1,
                "dna_motif_7mer": k7,
                "rna_motif_7mer": k7.replace("T","U"),
                "auc_drop": np.nan,
                "pos_fraction_in_top": np.nan,
                "lr_supporting_kmers": k
            })
            chosen.append(k7)
        if len(selected) >= 5:
            break

sel = pd.DataFrame(selected)
out_csv = OUTDIR / "selected_motifs_for_docking.csv"
sel.to_csv(out_csv, index=False)

# Write FASTA (RNA)
out_fa = OUTDIR / "selected_motifs_for_docking.fasta"
with open(out_fa, "w") as f:
    for i,row in sel.iterrows():
        name = f"{row['source']}_motif{i+1}_filter{row['filter']}_aucDrop{row['auc_drop']}"
        f.write(f">{name}\n{row['rna_motif_7mer']}\n")

print("Saved:", out_csv)
print("Saved:", out_fa)
print(sel.to_string(index=False))
