import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load prediction files
lr = pd.read_csv("lr_predictions.csv")
cnn = pd.read_csv("cnn_predictions.csv")

# Compute ROC for Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(lr["true_label"], lr["probability"])
auc_lr = auc(fpr_lr, tpr_lr)

# Compute ROC for CNN
fpr_cnn, tpr_cnn, _ = roc_curve(cnn["true_label"], cnn["probability"])
auc_cnn = auc(fpr_cnn, tpr_cnn)

# Plot
plt.figure(figsize=(6,6))

plt.plot(fpr_lr, tpr_lr, linewidth=2,
         label=f"Logistic Regression (AUC = {auc_lr:.3f})")

plt.plot(fpr_cnn, tpr_cnn, linewidth=2,
         label=f"CNN (AUC = {auc_cnn:.3f})")

plt.plot([0,1],[0,1],'k--', linewidth=1)

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve Comparison – TDP-43 Binding Prediction", fontsize=13)
plt.legend(loc="lower right")
plt.tight_layout()

# Save high-quality versions
plt.savefig("ROC_comparison.png", dpi=300)
plt.savefig("ROC_comparison.pdf")

plt.show()

print("Saved ROC_comparison.png and ROC_comparison.pdf")
