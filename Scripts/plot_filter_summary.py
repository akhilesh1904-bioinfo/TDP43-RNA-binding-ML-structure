import pandas as pd
import matplotlib.pyplot as plt

importance = pd.read_csv("interpret_cnn/filter_importance_fast.csv")
motifs = pd.read_csv("interpret_cnn/filter_motifs_summary_conv1.csv")

df = importance.merge(motifs, on="filter")

plt.figure(figsize=(8,6))

plt.scatter(
    df["auc_drop"],
    df["pos_fraction_in_top"],
    alpha=0.7
)

# label top filters
top = df.sort_values("auc_drop",ascending=False).head(10)

for _,row in top.iterrows():

    plt.text(
        row["auc_drop"],
        row["pos_fraction_in_top"],
        str(int(row["filter"]))
    )

plt.xlabel("Filter Importance (AUC drop)")
plt.ylabel("Motif Enrichment (Positive Fraction)")
plt.title("CNN Filter Importance vs Motif Enrichment")

plt.tight_layout()

plt.savefig(
    "interpret_cnn/figures/filter_importance_vs_motif_enrichment.png",
    dpi=300
)

print("Figure saved.")
