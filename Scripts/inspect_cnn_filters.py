import torch
import numpy as np
from train_cnn_tdp43 import TDP43CNN  # reuse same architecture


def load_model(weight_path="cnn_best.pt", seq_len=123):
    device = torch.device("cpu")
    model = TDP43CNN(seq_len=seq_len)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def filter_strength(filter_w):
    # Simple importance measure
    return np.sum(np.abs(filter_w))


def conv1_motifs(model, top_k=10):
    # conv1 weight shape: (n_filters, 4, kernel_size)
    with torch.no_grad():
        w = model.conv1.weight.detach().cpu().numpy()

    n_filters, n_channels, k = w.shape
    base_order = ["A", "C", "G", "T"]

    results = []

    for f_idx in range(n_filters):
        filt = w[f_idx]  # shape (4, k)
        strength = filter_strength(filt)

        motif_bases = []
        for pos in range(k):
            column = filt[:, pos]
            best_idx = int(np.argmax(column))
            motif_bases.append(base_order[best_idx])

        motif = "".join(motif_bases)
        results.append((f_idx, strength, motif))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def main():
    print("Loading best CNN model from cnn_best.pt ...")
    model = load_model("cnn_best.pt", seq_len=123)

    print("Extracting top conv1 filters...\n")
    top_filters = conv1_motifs(model, top_k=10)

    for f_idx, strength, motif in top_filters:
        print(f"Filter {f_idx}")
        print(f"  Strength: {strength:.3f}")
        print(f"  Motif (argmax bases): {motif}\n")


if __name__ == "__main__":
    main()
