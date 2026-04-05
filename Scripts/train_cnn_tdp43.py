import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score


RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ Data utilities ------------------ #

def read_fasta(path: str) -> List[str]:
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


def clean_seq(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join([c for c in seq if c in "ACGT"])


def one_hot_encode(seq: str, max_len: int) -> np.ndarray:
    """
    One-hot encode a DNA sequence to shape (4, max_len)

    A,C,G,T → channels 0,1,2,3
    Pad with zeros if shorter than max_len
    Truncate if longer
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, max_len), dtype=np.float32)

    seq = clean_seq(seq)
    for i, base in enumerate(seq[:max_len]):
        if base in mapping:
            arr[mapping[base], i] = 1.0
    return arr


class SeqDataset(Dataset):
    def __init__(self, seqs: List[str], labels: List[int], max_len: int):
        self.seqs = seqs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = one_hot_encode(self.seqs[idx], self.max_len)
        y = np.float32(self.labels[idx])
        return torch.tensor(x), torch.tensor(y)


def train_val_split(
    pos_seqs: List[str],
    neg_seqs: List[str],
    val_fraction: float = 0.2,
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    Balanced dataset: mix positives and negatives, then split into train/val.
    """
    n = min(len(pos_seqs), len(neg_seqs))
    random.seed(RANDOM_STATE)
    pos_seqs = random.sample(pos_seqs, n)
    neg_seqs = random.sample(neg_seqs, n)

    all_seqs = pos_seqs + neg_seqs
    labels = [1] * n + [0] * n

    # Shuffle together
    combined = list(zip(all_seqs, labels))
    random.shuffle(combined)
    all_seqs, labels = zip(*combined)
    all_seqs = list(all_seqs)
    labels = list(labels)

    split_idx = int(len(all_seqs) * (1 - val_fraction))
    train_seqs = all_seqs[:split_idx]
    train_labels = labels[:split_idx]
    val_seqs = all_seqs[split_idx:]
    val_labels = labels[split_idx:]

    return (train_seqs, train_labels), (val_seqs, val_labels)


# ------------------ CNN model ------------------ #

class TDP43CNN(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

        # in_channels = 4 (A,C,G,T), out_channels = 64
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=4)

        # We'll use global max pooling at the end
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch_size, 4, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Global max pooling over length dimension
        x = torch.max(x, dim=2).values  # shape: (batch_size, 128)

        logits = self.classifier(x)  # shape: (batch_size, 1)
        return logits.squeeze(1)


# ------------------ Training and evaluation ------------------ #

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc


# ------------------ Main script ------------------ #

def main():
    pos_fasta = "TARDBP_peaks.fa"
    neg_fasta = "TARDBP_negatives.fa"

    print("Reading sequences...")
    pos_raw = [clean_seq(s) for s in read_fasta(pos_fasta)]
    neg_raw = [clean_seq(s) for s in read_fasta(neg_fasta)]
    print(f"Loaded {len(pos_raw)} positives and {len(neg_raw)} negatives")

    # Choose a fixed length for CNN input (95th percentile, capped at 400)
    lengths = [len(s) for s in pos_raw + neg_raw]
    max_len = int(np.percentile(lengths, 95))
    max_len = min(max_len, 400)
    print(f"Using sequence length {max_len} (95th percentile, capped at 400)")

    (train_seqs, train_labels), (val_seqs, val_labels) = train_val_split(
        pos_raw, neg_raw, val_fraction=0.2
    )

    print(f"Train size: {len(train_seqs)}, Val size: {len(val_seqs)}")

    train_dataset = SeqDataset(train_seqs, train_labels, max_len=max_len)
    val_dataset = SeqDataset(val_seqs, val_labels, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = TDP43CNN(seq_len=max_len).to(DEVICE)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 10
    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_acc, val_auc = eval_epoch(
            model, val_loader, criterion
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.3f}, auc {train_auc:.3f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.3f}, auc {val_auc:.3f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()

    if best_state is not None:
        # save best model weights to file
        torch.save(best_state, "cnn_best.pt")
        model.load_state_dict(best_state)

    print("\n=== Best CNN Validation Performance ===")
    print(f"Best val ROC AUC: {best_val_auc:.3f}")
    # ---- Export validation predictions ----
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())

    import pandas as pd
    df = pd.DataFrame({
        "true_label": all_labels,
        "probability": all_probs
    })
    df.to_csv("cnn_predictions.csv", index=False)
    print("Saved cnn_predictions.csv")


if __name__ == "__main__":
    main()
