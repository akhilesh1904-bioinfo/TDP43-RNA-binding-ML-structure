import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

MODEL_PATH = "cnn_best.pt"
DATA_CSV = "tdp43_sequences_labeled.csv"

device = "cpu"

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(4,64,kernel_size=7)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64,128,kernel_size=7)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.gpool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1)
        )

    def forward(self,x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.gpool(x)
        x = x.flatten(1)

        x = self.classifier(x)

        return x


state_dict = torch.load(MODEL_PATH,map_location="cpu")
model = CNNModel()

# match classifier dims
w0 = state_dict["classifier.0.weight"]
w3 = state_dict["classifier.3.weight"]

model.classifier = nn.Sequential(
    nn.Linear(w0.shape[1],w0.shape[0]),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(w3.shape[1],w3.shape[0])
)

model.load_state_dict(state_dict)
model.eval()


print("Model loaded")

# ---------- Load sequences ----------

df = pd.read_csv(DATA_CSV)

seqs = df["sequence"].tolist()
labels = df["label"].values


BASES = "ACGT"
b2i = {b:i for i,b in enumerate(BASES)}

def one_hot(seq):

    L = len(seq)
    arr = np.zeros((4,L))

    for i,ch in enumerate(seq):

        if ch in b2i:
            arr[b2i[ch],i] = 1

    return arr


def predict(seqs):

    preds = []

    for seq in seqs:

        x = one_hot(seq)

        x = torch.tensor(x).unsqueeze(0).float()

        with torch.no_grad():

            p = torch.sigmoid(model(x)).item()

        preds.append(p)

    return np.array(preds)


print("Computing baseline predictions")

baseline = predict(seqs)

baseline_auc = np.mean((baseline > 0.5) == labels)

print("Baseline accuracy:",baseline_auc)

# ---------- Filter ablation ----------

importance = []

for f in tqdm(range(64)):

    # backup weights
    w_backup = model.conv1.weight.data[f].clone()

    # zero filter
    model.conv1.weight.data[f] = 0

    preds = predict(seqs)

    acc = np.mean((preds > 0.5) == labels)

    drop = baseline_auc - acc

    importance.append((f,drop))

    # restore weights
    model.conv1.weight.data[f] = w_backup


imp_df = pd.DataFrame(importance,columns=["filter","accuracy_drop"])

imp_df = imp_df.sort_values("accuracy_drop",ascending=False)

imp_df.to_csv("interpret_cnn/filter_importance.csv",index=False)

print("\nTop important filters")

print(imp_df.head(10))
