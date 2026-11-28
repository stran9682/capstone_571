"""
SVM model for Team 2's capstone: What combination of factors causes the most stress?

Target: stress_level (0 = low, 1 = medium, 2 = high)
Features: the 20 other columns in StressLevelDataset.csv

Outputs
  results/svm_metrics.csv           - accuracy + macro F1 (same columns teammates can use)
  results/svm_best_params.csv       - best kernel / C / gamma from the grid search
  results/svm_classification_report.txt
  results/svm_permutation_importance.csv
  results/svm_feature_pairs.csv     - how well each PAIR of top factors predicts stress on its own
  figures/svm_confusion_matrix.png
  figures/svm_permutation_importance.png
  figures/svm_feature_pairs.png
"""

import os
import sqlite3
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 1. Load data
data = pd.read_csv("data/StressLevelDataset.csv")
con = sqlite3.connect(":memory:")
data.to_sql("data", con, index=False, if_exists="replace")


def q(sql):
    return pd.read_sql(sql, con)


FEATURES = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
]
TARGET = "stress_level"

df = q(f"""
    SELECT      {", ".join(FEATURES)},
                {TARGET}
    FROM        data
""")

print("Original rows:", len(df))
df = df.dropna()  # drop on features AND target together so rows stay aligned
print("Rows after filtering missing values:", len(df))
print("Class balance:\n", df[TARGET].value_counts().sort_index(), "\n")

X = df[FEATURES]
y = df[TARGET].astype(int)

# 2. Train / test split (stratified so each stress level is represented evenly)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 3. SVM pipeline + grid search
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC()),
])

param_grid = [
    {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10]},
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    },
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)
print(f"Best CV macro F1: {grid.best_score_:.3f}\n")

pd.DataFrame([{**grid.best_params_, "cv_macro_f1": grid.best_score_}]).to_csv(
    "results/svm_best_params.csv", index=False
)

# 4. Evaluate on the held-out test set
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
report = classification_report(
    y_test, y_pred, target_names=["Low (0)", "Medium (1)", "High (2)"]
)

print(f"Test accuracy: {acc:.3f}")
print(f"Test macro F1: {macro_f1:.3f}")
print(report)

pd.DataFrame([{"model": "SVM", "accuracy": acc, "macro_f1": macro_f1}]).to_csv(
    "results/svm_metrics.csv", index=False
)
with open("results/svm_classification_report.txt", "w") as f:
    f.write(f"Best params: {grid.best_params_}\n\n{report}")

# Shared styling for figures
BLUE = "#257de0"
TEXT = "#191919"
TEXT_2 = "#666666"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cce3fd", "#84b9f3", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
)
plt.rcParams.update({
    "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Low", "Medium", "High"]).plot(
    ax=ax, cmap=BLUE_RAMP, colorbar=False
)
ax.set_title("SVM confusion matrix (test set)", color=TEXT)
ax.set_xlabel("Predicted stress level")
ax.set_ylabel("Actual stress level")
fig.tight_layout()
fig.savefig("figures/svm_confusion_matrix.png", dpi=200)
plt.close(fig)

# 5. Which factors matter most? (permutation importance)
perm = permutation_importance(
    best_model, X_test, y_test,
    scoring="f1_macro", n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1,
)
importance = (
    pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    })
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)
importance.to_csv("results/svm_permutation_importance.csv", index=False)
print("Permutation importance:\n", importance, "\n")

plot_df = importance.iloc[::-1]  # largest at the top
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.barh(
    plot_df["feature"], plot_df["importance_mean"],
    xerr=plot_df["importance_std"], color=BLUE, height=0.6,
    error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 2},
)
ax.axvline(0, color=TEXT_2, linewidth=0.8)
ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("Drop in macro F1 when the feature is shuffled")
ax.set_title("SVM: which factors matter most", color=TEXT, loc="left")
fig.tight_layout()
fig.savefig("figures/svm_permutation_importance.png", dpi=200)
plt.close(fig)

# 6. Which COMBINATIONS matter most? (pairs of top factors)
TOP_K = 6
top_features = importance["feature"].head(TOP_K).tolist()

pair_rows = []
for f1_name, f2_name in combinations(top_features, 2):
    pair_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**{k.replace("svc__", ""): v for k, v in grid.best_params_.items()})),
    ])
    scores = cross_val_score(
        pair_pipe, X_train[[f1_name, f2_name]], y_train, cv=cv, scoring="f1_macro"
    )
    pair_rows.append({
        "feature_1": f1_name,
        "feature_2": f2_name,
        "cv_macro_f1_mean": scores.mean(),
        "cv_macro_f1_std": scores.std(),
    })

pairs = (
    pd.DataFrame(pair_rows)
    .sort_values("cv_macro_f1_mean", ascending=False)
    .reset_index(drop=True)
)
pairs.to_csv("results/svm_feature_pairs.csv", index=False)
print(f"Best pairs among top {TOP_K} features:\n", pairs.head(10), "\n")

# Heatmap of pair scores
grid_scores = pd.DataFrame(np.nan, index=top_features, columns=top_features)
for _, r in pairs.iterrows():
    grid_scores.loc[r.feature_1, r.feature_2] = r.cv_macro_f1_mean
    grid_scores.loc[r.feature_2, r.feature_1] = r.cv_macro_f1_mean

fig, ax = plt.subplots(figsize=(7, 6))
masked = np.ma.masked_invalid(grid_scores.values)
im = ax.imshow(masked, cmap=BLUE_RAMP)
ax.set_xticks(range(TOP_K), top_features, rotation=45, ha="right")
ax.set_yticks(range(TOP_K), top_features)
ax.spines[:].set_visible(False)
vmin, vmax = np.nanmin(grid_scores.values), np.nanmax(grid_scores.values)
for i in range(TOP_K):
    for j in range(TOP_K):
        val = grid_scores.values[i, j]
        if not np.isnan(val):
            dark_cell = (val - vmin) / (vmax - vmin + 1e-9) > 0.55
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark_cell else TEXT)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("CV macro F1 using only this pair")
ax.set_title("SVM: how well each pair of factors predicts stress", color=TEXT, loc="left")
fig.tight_layout()
fig.savefig("figures/svm_feature_pairs.png", dpi=200)
plt.close(fig)

print("Done. Results in results/, figures in figures/.")
