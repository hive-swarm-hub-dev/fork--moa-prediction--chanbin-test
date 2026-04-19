import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
train_features = pd.read_csv("data/train_features.csv")
train_targets = pd.read_csv("data/train_targets.csv")
test_features = pd.read_csv("data/test_features.csv")

target_cols = [c for c in train_targets.columns if c != "sig_id"]

# Separate control samples
train_ctrl_mask = train_features["cp_type"] == "ctl_vehicle"
test_ctrl_mask = test_features["cp_type"] == "ctl_vehicle"

# Encode categoricals
all_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)
all_features["cp_type"] = LabelEncoder().fit_transform(all_features["cp_type"])
all_features["cp_dose"] = LabelEncoder().fit_transform(all_features["cp_dose"])

gene_cols = [c for c in all_features.columns if c.startswith("g-")]
cell_cols = [c for c in all_features.columns if c.startswith("c-")]

# Standardize before PCA
gene_scaler = StandardScaler()
cell_scaler = StandardScaler()
gene_scaled = gene_scaler.fit_transform(all_features[gene_cols])
cell_scaled = cell_scaler.fit_transform(all_features[cell_cols])

# PCA — more components to capture more signal
pca_gene = PCA(n_components=100, random_state=42)
gene_pca = pca_gene.fit_transform(gene_scaled)

pca_cell = PCA(n_components=30, random_state=42)
cell_pca = pca_cell.fit_transform(cell_scaled)

# Also add variance/stats features
gene_var = np.var(gene_scaled, axis=1, keepdims=True)
gene_mean = np.mean(gene_scaled, axis=1, keepdims=True)
cell_var = np.var(cell_scaled, axis=1, keepdims=True)
cell_mean = np.mean(cell_scaled, axis=1, keepdims=True)

# Build feature matrix
X_all = np.hstack([
    all_features[["cp_type", "cp_time", "cp_dose"]].values,
    gene_pca,
    cell_pca,
    gene_var, gene_mean,
    cell_var, cell_mean,
])

# Final scaling
final_scaler = StandardScaler()
X_all = final_scaler.fit_transform(X_all)

n_train = len(train_features)
X_train = X_all[:n_train]
X_test = X_all[n_train:]
y_train = train_targets[target_cols].values

# Train on treatment samples only
train_trt_idx = np.where(~train_ctrl_mask.values)[0]
X_train_trt = X_train[train_trt_idx]
y_train_trt = y_train[train_trt_idx]

# Logistic regression with tuned regularization
model = OneVsRestClassifier(
    LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"),
    n_jobs=-1,
)
model.fit(X_train_trt, y_train_trt)
test_preds = model.predict_proba(X_test)

# Clip predictions
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)

# Zero out control samples
test_ctrl_indices = np.where(test_ctrl_mask.values)[0]
test_preds[test_ctrl_indices] = 1e-15

# Build submission
submission = pd.DataFrame(test_preds, columns=target_cols)
submission.insert(0, "sig_id", test_features["sig_id"].values)
submission.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows x {len(target_cols)} targets")
