import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

summary = pd.read_csv("interpret_cnn/filter_motifs_summary_conv1.csv")
pwms = np.load("interpret_cnn/arrays/conv1_pwms_all.npy")

top = summary.sort_values("pos_fraction_in_top", ascending=False).head(8)

bases = ["A","C","G","T"]

for _,row in top.iterrows():

    f = int(row["filter"])
    pwm = pwms[f]

    plt.figure(figsize=(6,2))
    plt.imshow(pwm, aspect="auto")

    plt.yticks(range(4), bases)
    plt.xticks(range(7), range(1,8))

    plt.title(f"Filter {f} motif: {row['consensus_all']}")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"interpret_cnn/figures/filter_{f}_motif.png", dpi=300)
    plt.close()

print("Motif plots saved.")
