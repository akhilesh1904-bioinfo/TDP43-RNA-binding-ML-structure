import argparse
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

RANDOM_STATE = 42


def read_fasta(path):
    """Read sequences from a FASTA file into a list of strings."""
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


def seq_to_kmers(seq, k):
    """Return list of all k-length substrings from a sequence."""
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def seq_to_token_string(seq, ks=(2, 3)):
    """
    Convert a sequence into a whitespace-separated string of k-mers.

    Example output: "UG UG GG GT GT ..."
    (Here we convert U->T and keep only A/C/G/T)
    """
    seq = seq.upper().replace("U", "T")
    seq = "".join([c for c in seq if c in "ACGT"])
    tokens = []
    for k in ks:
        if len(seq) >= k:
            tokens.extend(seq_to_kmers(seq, k))
    return " ".join(tokens)


def main(pos_fasta, neg_fasta, ks=(2, 3), test_size=0.2):
    print(f"Reading positives from {pos_fasta}")
    pos_seqs = read_fasta(pos_fasta)

    print(f"Reading negatives from {neg_fasta}")
    neg_seqs = read_fasta(neg_fasta)

    print(f"Loaded {len(pos_seqs)} positives and {len(neg_seqs)} negatives")

    # Balance the dataset in case one side is larger
    random.seed(RANDOM_STATE)
    n = min(len(pos_seqs), len(neg_seqs))
    pos_seqs = random.sample(pos_seqs, n)
    neg_seqs = random.sample(neg_seqs, n)
    print(f"Balanced to {n} positives and {n} negatives (total {2 * n})")

    all_seqs = pos_seqs + neg_seqs
    y = np.array([1] * n + [0] * n)

    # Convert sequences to k-mer "documents"
    print(f"Converting sequences to k-mer tokens with ks={ks} ...")
    texts = [seq_to_token_string(s, ks=ks) for s in all_seqs]
    print("Example tokenized sequence:")
    print(texts[0][:120] + " ...")

    # Vectorize k-mers (bag-of-words over k-mers)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    print(f"Feature matrix shape: {X.shape} (n_samples, n_features)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Logistic regression on k-mers
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== k-mer Logistic Regression Results ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC:  {auc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Interpret top k-mers (motifs)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_.ravel()

    top_pos_idx = np.argsort(coefs)[-20:][::-1]  # largest positive coefficients
    top_neg_idx = np.argsort(coefs)[:20]         # most negative coefficients

    print("\nTop 20 enriched k-mers (positive TDP-43 class):")
    for idx in top_pos_idx:
        print(f"{feature_names[idx]:>6}  coef={coefs[idx]:.3f}")

    print("\nTop 20 depleted k-mers (negative/non-binding class):")
    for idx in top_neg_idx:
        print(f"{feature_names[idx]:>6}  coef={coefs[idx]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDP-43 k-mer logistic regression")
    parser.add_argument(
        "--pos_fasta",
        type=str,
        default="TARDBP_peaks.fa",
        help="FASTA with positive binding sequences",
    )
    parser.add_argument(
        "--neg_fasta",
        type=str,
        default="TARDBP_negatives.fa",
        help="FASTA with negative/non-binding sequences",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="2,3",
        help="Comma-separated k-mer sizes, e.g. '2,3' or '3,4'",
    )
    args = parser.parse_args()

    ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    main(args.pos_fasta, args.neg_fasta, ks=ks)
