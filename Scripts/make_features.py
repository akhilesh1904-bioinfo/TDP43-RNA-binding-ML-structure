import csv

def gc_content(seq):
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)

def g_fraction(seq):
    return seq.count("G") / len(seq)

def t_fraction(seq):
    return seq.count("T") / len(seq)

def gt_dinuc_fraction(seq):
    count = 0
    for i in range(len(seq) - 1):
        if seq[i:i+2] == "GT":
            count += 1
    return count / (len(seq) - 1)

input_file = "tdp43_sequences_labeled.csv"
output_file = "tdp43_basic_features.csv"

rows_out = []

with open(input_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        seq = row["sequence"]
        features = {
            "id": row["id"],
            "gc_content": gc_content(seq),
            "g_fraction": g_fraction(seq),
            "t_fraction": t_fraction(seq),
            "gt_dinuc_fraction": gt_dinuc_fraction(seq),
            "length": len(seq),
            "label": int(row["label"])
        }
        rows_out.append(features)

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Wrote {len(rows_out)} rows to {output_file}")
