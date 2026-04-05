import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def read_fasta(path):
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            sequences.append("".join(current))
    return sequences

def clean_seq(seq):
    return "".join([c for c in seq.upper().replace("U","T") if c in "ACGT"])

def seq_to_kmers(seq, ks=(2,3)):
    tokens = []
    for k in ks:
        for i in range(len(seq)-k+1):
            tokens.append(seq[i:i+k])
    return " ".join(tokens)

print("Reading sequences...")
pos = [clean_seq(s) for s in read_fasta("TARDBP_peaks.fa")]
neg = [clean_seq(s) for s in read_fasta("TARDBP_negatives.fa")]

n = min(len(pos), len(neg))
random.seed(RANDOM_STATE)
pos = random.sample(pos, n)
neg = random.sample(neg, n)

all_seqs = pos + neg
y = np.array([1]*n + [0]*n)

texts = [seq_to_kmers(s, ks=(2,3)) for s in all_seqs]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

feature_names = np.array(vectorizer.get_feature_names_out())
coefs = clf.coef_.ravel()

# Top 15 enriched for positive class
top_idx = np.argsort(coefs)[-15:][::-1]

df = pd.DataFrame({
    "kmer": feature_names[top_idx],
    "coefficient": coefs[top_idx]
})

df.to_csv("top_kmers.csv", index=False)
print("Saved top_kmers.csv")
