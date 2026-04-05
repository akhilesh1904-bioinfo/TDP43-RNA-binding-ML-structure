import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "cnn_best.pt"

print("Loading CNN model weights...")

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7)
        self.bn2   = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

        # global pooling to get (N, 128, 1) -> flatten => 128
        self.gpool = nn.AdaptiveMaxPool1d(1)

        # classifier matches keys classifier.0 and classifier.3
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),   # out_features will be validated by state_dict
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.gpool(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


state_dict = torch.load(MODEL_PATH, map_location="cpu")

# Adjust classifier layer sizes automatically from weights (no guessing)
w0 = state_dict["classifier.0.weight"]   # (out0, 128)
b0 = state_dict["classifier.0.bias"]
w3 = state_dict["classifier.3.weight"]   # (out3, in3)
b3 = state_dict["classifier.3.bias"]

model = CNNModel()

# rebuild classifier to exactly match saved dims
model.classifier = nn.Sequential(
    nn.Linear(w0.shape[1], w0.shape[0]),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(w3.shape[1], w3.shape[0]),
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")
print("conv1:", model.conv1)
print("conv2:", model.conv2)
print("Kernel size:", model.conv1.kernel_size, "Filters:", model.conv1.out_channels)
print("Classifier:", model.classifier)
