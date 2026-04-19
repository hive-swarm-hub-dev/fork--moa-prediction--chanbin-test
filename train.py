import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

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

# Standardize
scaler = StandardScaler()
numeric_data = scaler.fit_transform(all_features[gene_cols + cell_cols])

# PCA — push even higher
pca_gene = PCA(n_components=200, random_state=42)
gene_pca = pca_gene.fit_transform(numeric_data[:, :len(gene_cols)])

pca_cell = PCA(n_components=60, random_state=42)
cell_pca = pca_cell.fit_transform(numeric_data[:, len(gene_cols):])

# Stats features
gene_data = numeric_data[:, :len(gene_cols)]
cell_data = numeric_data[:, len(gene_cols):]
gene_var = np.var(gene_data, axis=1, keepdims=True)
gene_skew = np.mean(gene_data**3, axis=1, keepdims=True)
cell_var = np.var(cell_data, axis=1, keepdims=True)

# Build feature matrix
X_all = np.hstack([
    all_features[["cp_type", "cp_time", "cp_dose"]].values,
    gene_pca,
    cell_pca,
    gene_var, gene_skew, cell_var,
])

final_scaler = StandardScaler()
X_all = final_scaler.fit_transform(X_all)

n_train = len(train_features)
X_train = X_all[:n_train]
X_test = X_all[n_train:]
y_train = train_targets[target_cols].values

# Treatment samples only
train_trt_idx = np.where(~train_ctrl_mask.values)[0]
X_trt = X_train[train_trt_idx]
y_trt = y_train[train_trt_idx]

# Multi-seed MLP ensemble to reduce variance
mlp_preds_list = []
for seed in [42, 123, 777]:
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=seed,
        verbose=False,
    )
    mlp.fit(X_trt, y_trt)
    mlp_preds_list.append(mlp.predict_proba(X_test))
    print(f"MLP seed={seed} done")

mlp_preds = np.mean(mlp_preds_list, axis=0)

# LogReg
lr = OneVsRestClassifier(
    LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"),
    n_jobs=-1,
)
lr.fit(X_trt, y_trt)
lr_preds = lr.predict_proba(X_test)
print("LogReg done")

# Blend
test_preds = 0.7 * mlp_preds + 0.3 * lr_preds

# Clip and zero controls
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_ctrl_indices = np.where(test_ctrl_mask.values)[0]
test_preds[test_ctrl_indices] = 1e-15

# Build submission
submission = pd.DataFrame(test_preds, columns=target_cols)
submission.insert(0, "sig_id", test_features["sig_id"].values)
submission.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows x {len(target_cols)} targets")
