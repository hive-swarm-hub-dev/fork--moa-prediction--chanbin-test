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

# PCA at two scales
pca_gene = PCA(n_components=200, random_state=42)
gene_pca = pca_gene.fit_transform(numeric_data[:, :len(gene_cols)])
pca_cell = PCA(n_components=60, random_state=42)
cell_pca = pca_cell.fit_transform(numeric_data[:, len(gene_cols):])

# Second PCA on combined features (captures cross-modality patterns)
pca_combined = PCA(n_components=50, random_state=42)
combined_pca = pca_combined.fit_transform(numeric_data)

# Stats
gene_data = numeric_data[:, :len(gene_cols)]
cell_data = numeric_data[:, len(gene_cols):]
gene_var = np.var(gene_data, axis=1, keepdims=True)
gene_skew = np.mean(gene_data**3, axis=1, keepdims=True)
gene_kurt = np.mean(gene_data**4, axis=1, keepdims=True) - 3
cell_var = np.var(cell_data, axis=1, keepdims=True)
cell_skew = np.mean(cell_data**3, axis=1, keepdims=True)

# Percentile features
gene_q25 = np.percentile(gene_data, 25, axis=1, keepdims=True)
gene_q75 = np.percentile(gene_data, 75, axis=1, keepdims=True)
gene_iqr = gene_q75 - gene_q25
cell_q25 = np.percentile(cell_data, 25, axis=1, keepdims=True)
cell_q75 = np.percentile(cell_data, 75, axis=1, keepdims=True)
cell_iqr = cell_q75 - cell_q25

# Interaction features
cp_time = all_features["cp_time"].values.reshape(-1, 1) / 72.0
cp_dose = all_features["cp_dose"].values.reshape(-1, 1)

gene_time_interact = gene_pca[:, :20] * cp_time
gene_dose_interact = gene_pca[:, :20] * cp_dose
cell_time_interact = cell_pca[:, :10] * cp_time
cell_dose_interact = cell_pca[:, :10] * cp_dose
cross_pca = gene_pca[:, :5] * cell_pca[:, :5]
gene_pca_sq = gene_pca[:, :10] ** 2

X_all = np.hstack([
    all_features[["cp_type", "cp_time", "cp_dose"]].values,
    gene_pca, cell_pca, combined_pca,
    gene_var, gene_skew, gene_kurt, gene_iqr,
    cell_var, cell_skew, cell_iqr,
    gene_time_interact, gene_dose_interact,
    cell_time_interact, cell_dose_interact,
    cross_pca, gene_pca_sq,
])

final_scaler = StandardScaler()
X_all = final_scaler.fit_transform(X_all)

n_train = len(train_features)
X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train = train_targets[target_cols].values

trt_idx = np.where(~train_ctrl_mask.values)[0]
X_trt, y_trt = X_train[trt_idx], y_train[trt_idx]

# Diverse MLP ensemble — wider networks, lower alpha
configs = [
    ((768, 384), 0.0003, 42),
    ((768, 384), 0.0003, 123),
    ((512, 256), 0.0005, 777),
    ((512, 256, 128), 0.0008, 42),
]

mlp_preds_list = []
for layers, alpha, seed in configs:
    mlp = MLPClassifier(
        hidden_layer_sizes=layers, activation="relu", solver="adam",
        alpha=alpha, batch_size=256, learning_rate="adaptive",
        learning_rate_init=0.001, max_iter=300, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15,
        random_state=seed, verbose=False,
    )
    mlp.fit(X_trt, y_trt)
    mlp_preds_list.append(mlp.predict_proba(X_test))
    print(f"MLP {layers} seed={seed} done")

mlp_preds = np.mean(mlp_preds_list, axis=0)

# Multi-C LogReg
lr_preds_list = []
for c_val in [0.05, 0.1, 0.2]:
    lr = OneVsRestClassifier(
        LogisticRegression(C=c_val, max_iter=2000, solver="lbfgs"), n_jobs=-1,
    )
    lr.fit(X_trt, y_trt)
    lr_preds_list.append(lr.predict_proba(X_test))
    print(f"LogReg C={c_val} done")

lr_preds = np.mean(lr_preds_list, axis=0)

# Blend — MLP-heavy since it's stronger
test_preds = 0.75 * mlp_preds + 0.25 * lr_preds

# Post-processing: calibrate using target priors
# Scale predictions toward base rate with a soft blend
target_means = y_trt.mean(axis=0)
# Gentle calibration: shrink extreme predictions slightly toward prior
calibration_strength = 0.05
for i in range(len(target_cols)):
    prior = target_means[i]
    test_preds[:, i] = (1 - calibration_strength) * test_preds[:, i] + calibration_strength * prior

# Clip and zero controls
test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
test_ctrl_indices = np.where(test_ctrl_mask.values)[0]
test_preds[test_ctrl_indices] = 1e-15

submission = pd.DataFrame(test_preds, columns=target_cols)
submission.insert(0, "sig_id", test_features["sig_id"].values)
submission.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows x {len(target_cols)} targets")
