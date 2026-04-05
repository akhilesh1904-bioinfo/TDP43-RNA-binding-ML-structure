import csv
import random

random.seed(42)  # for reproducibility


def read_fasta(path):
    """Return list of (name, sequence) from a FASTA file."""
    records = []
    name = None
    seq_chunks = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # save previous record
                if name is not None:
                    records.append((name, "".join(seq_chunks).upper()))
                name = line[1:]  # drop ">"
                seq_chunks = []
            else:
                seq_chunks.append(line)
        # last record
        if name is not None:
            records.append((name, "".join(seq_chunks).upper()))
    return records


# Read positives and negatives
pos_fasta = "TARDBP_peaks.fa"
neg_fasta = "TARDBP_negatives.fa"

positives = read_fasta(pos_fasta)
negatives = read_fasta(neg_fasta)

print(f"Loaded {len(positives)} positives and {len(negatives)} negatives")

# Optional: downsample negatives to match positives
if len(negatives) > len(positives):
    negatives = random.sample(negatives, len(positives))
    print(f"Downsampled negatives to {len(negatives)} to balance classes")

rows = []

# Add positives with label 1
for i, (name, seq) in enumerate(positives, start=1):
    rows.append({
        "id": f"pos_{i:06d}",
        "source_name": name,
        "sequence": seq,
        "label": 1
    })

# Add negatives with label 0
for i, (name, seq) in enumerate(negatives, start=1):
    rows.append({
        "id": f"neg_{i:06d}",
        "source_name": name,
        "sequence": seq,
        "label": 0
    })

# Shuffle rows so positives/negatives are mixed
random.shuffle(rows)

out_path = "tdp43_sequences_labeled.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "source_name", "sequence", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} sequences to {out_path}")
