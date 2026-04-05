import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv("tdp43_basic_features.csv")

feature_cols = ["gc_content", "g_fraction", "t_fraction", "gt_dinuc_fraction", "length"]
X = df[feature_cols]
y = df["label"]

print(f"Total samples: {len(df)}")
print("Class counts:")
print(y.value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC: {roc_auc:.3f}")

# Coefficients
clf = pipe.named_steps["clf"]

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, clf.coef_[0]):
    print(f"{name:20s} {coef:+.3f}")
