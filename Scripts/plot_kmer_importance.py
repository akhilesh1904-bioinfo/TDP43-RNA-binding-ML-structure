import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("top_kmers.csv")

plt.figure(figsize=(8,5))
plt.barh(df["kmer"][::-1], df["coefficient"][::-1])
plt.xlabel("Logistic Regression Coefficient")
plt.title("Top Enriched k-mers for TDP-43 Binding")
plt.tight_layout()

plt.savefig("kmer_importance.png", dpi=300)
plt.savefig("kmer_importance.pdf")

plt.show()

print("Saved kmer_importance.png and kmer_importance.pdf")
