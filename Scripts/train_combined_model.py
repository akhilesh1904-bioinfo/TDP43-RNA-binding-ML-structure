import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

RANDOM_STATE = 42


def read_fasta(path):
    sequences = []
    current = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
    seq = seq.upper().replace("U", "T")
    return "".join([c for c in seq if c in "ACGT"])


def basic_features(seq):
    length = len(seq)
    if length == 0:
        return [0, 0, 0, 0, 0]

    g_frac = seq.count("G") / length
    t_frac = seq.count("T") / length
    gc_frac = (seq.count("G") + seq.count("C")) / length
    gt_freq = sum(1 for i in range(len(seq)-1) if seq[i:i+2] == "GT") / length

    return [g_frac, t_frac, gc_frac, gt_freq, length]


def seq_to_kmers(seq, ks=(2, 3)):
    tokens = []
    for k in ks:
        for i in range(len(seq) - k + 1):
            tokens.append(seq[i:i+k])
    return " ".join(tokens)


def main():
    pos_fasta = "TARDBP_peaks.fa"
    neg_fasta = "TARDBP_negatives.fa"

    print("Reading sequences...")
    pos_seqs = [clean_seq(s) for s in read_fasta(pos_fasta)]
    neg_seqs = [clean_seq(s) for s in read_fasta(neg_fasta)]

    n = min(len(pos_seqs), len(neg_seqs))
    random.seed(RANDOM_STATE)
    pos_seqs = random.sample(pos_seqs, n)
    neg_seqs = random.sample(neg_seqs, n)

    all_seqs = pos_seqs + neg_seqs
    y = np.array([1]*n + [0]*n)

    print(f"Balanced dataset: {2*n} sequences")

    # ----- Basic features -----
    basic_X = np.array([basic_features(seq) for seq in all_seqs])

    # ----- k-mer features (2+3) -----
    texts = [seq_to_kmers(seq, ks=(2, 3)) for seq in all_seqs]
    vectorizer = CountVectorizer()
    kmer_X = vectorizer.fit_transform(texts).toarray()

    print("Basic feature shape:", basic_X.shape)
    print("k-mer feature shape:", kmer_X.shape)

    # ----- Combine features -----
    X = np.hstack([basic_X, kmer_X])
    print("Combined feature shape:", X.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== Combined Feature Model Results ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC:  {auc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
