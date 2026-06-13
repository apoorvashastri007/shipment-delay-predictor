"""
Shipment Delay Prediction - Full ML Pipeline
Includes: preprocessing, feature engineering, model training, evaluation, and export.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("SHIPMENT DELAY PREDICTION – ML PIPELINE")
print("=" * 60)

df = pd.read_csv("data/shipments.csv")
print(f"\n✓ Loaded {len(df):,} records | Delay rate: {df['is_delayed'].mean():.1%}\n")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df["ship_date"] = pd.to_datetime(df["ship_date"])
df["ship_month"]   = df["ship_date"].dt.month
df["ship_quarter"] = df["ship_date"].dt.quarter
df["ship_dayofweek"] = df["ship_date"].dt.dayofweek
df["is_weekend_departure"] = (df["ship_dayofweek"] >= 5).astype(int)

df["congestion_x_stops"]   = df["port_congestion_score"] * df["num_stops"]
df["weight_per_stop"]      = df["cargo_weight_tons"] / (df["num_stops"] + 1)
df["maintenance_risk"]     = (df["days_since_maintenance"] / 180).clip(0, 1)
df["carrier_risk_score"]   = 1 - df["carrier_on_time_rate"]

WEATHER_SEVERITY = {
    "Clear": 0, "Cloudy": 1, "Rainy": 2, "Storm": 3, "Hurricane": 4
}
df["weather_severity"] = df["weather_condition"].map(WEATHER_SEVERITY)
df["high_risk_weather"]    = (df["weather_severity"] >= 3).astype(int)

df["route"]  = df["origin_port"] + " → " + df["destination_port"]
df["log_weight"] = np.log1p(df["cargo_weight_tons"])

print("✓ Feature engineering complete")

# ─────────────────────────────────────────────
# 3. DEFINE FEATURE SETS
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "base_transit_days", "port_congestion_score", "customs_complexity",
    "cargo_weight_tons", "num_stops", "is_peak_season",
    "carrier_on_time_rate", "days_since_maintenance", "fuel_cost_index",
    "ship_month", "ship_quarter", "ship_dayofweek", "is_weekend_departure",
    "congestion_x_stops", "weight_per_stop", "maintenance_risk",
    "carrier_risk_score", "weather_severity", "high_risk_weather", "log_weight",
]

CATEGORICAL_FEATURES = ["carrier", "weather_condition", "origin_port", "destination_port"]

TARGET = "is_delayed"
DROP_COLS = ["shipment_id", "ship_date", "actual_transit_days", "route", TARGET]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ─────────────────────────────────────────────
# 5. MODELS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12,
                                                  min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                      max_depth=5, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n── Cross-Validation Results ──────────────────────────")
cv_results = {}
for name, clf in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<25} AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 6. ENSEMBLE (VOTING)
# ─────────────────────────────────────────────
ensemble = VotingClassifier(
    estimators=[(n, Pipeline([("pre", preprocessor), ("clf", m)])) for n, m in models.items()],
    voting="soft",
)
ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"  {'Ensemble (Voting)':<25} AUC: {ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")

# ─────────────────────────────────────────────
# 7. TRAIN BEST MODEL (Gradient Boosting)
# ─────────────────────────────────────────────
best_name = "Gradient Boosting"
best_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", models[best_name]),
])
best_pipe.fit(X_train, y_train)
print(f"\n✓ Best model trained: {best_name}")

# ─────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────
y_pred      = best_pipe.predict(X_test)
y_prob      = best_pipe.predict_proba(X_test)[:, 1]
roc_auc     = roc_auc_score(y_test, y_prob)
avg_prec    = average_precision_score(y_test, y_prob)

print("\n── Test Set Metrics ──────────────────────────────────")
print(f"  ROC-AUC:           {roc_auc:.4f}")
print(f"  Avg Precision:     {avg_prec:.4f}")
print("\n" + classification_report(y_test, y_prob.round(), target_names=["On-Time", "Delayed"]))

# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
ohe_cats   = best_pipe.named_steps["pre"].named_transformers_["cat"]["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
feat_names = NUMERIC_FEATURES + list(ohe_cats)
importances = best_pipe.named_steps["clf"].feature_importances_
fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False).head(15)

# ─────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Shipment Delay Prediction – Model Evaluation", fontsize=16, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# (a) ROC Curve
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax1.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0,1],[0,1],"--", color="gray", alpha=0.5)
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve"); ax1.legend(loc="lower right")

# (b) PR Curve
ax2 = fig.add_subplot(gs[0, 1])
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ax2.plot(rec, prec, color="#16a34a", lw=2, label=f"AP = {avg_prec:.3f}")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve"); ax2.legend()

# (c) Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["On-Time", "Delayed"])
disp.plot(ax=ax3, colorbar=False, cmap="Blues")
ax3.set_title("Confusion Matrix")

# (d) Feature Importance
ax4 = fig.add_subplot(gs[1, :2])
colors = ["#2563eb" if i < 5 else "#93c5fd" for i in range(len(fi_df))]
ax4.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors[::-1])
ax4.set_xlabel("Importance"); ax4.set_title("Top 15 Feature Importances")

# (e) CV AUC Distribution
ax5 = fig.add_subplot(gs[1, 2])
all_names  = list(cv_results.keys()) + ["Ensemble"]
all_scores = list(cv_results.values()) + [ensemble_scores]
bp = ax5.boxplot(all_scores, patch_artist=True, labels=[n.replace(" ", "\n") for n in all_names])
for patch in bp["boxes"]:
    patch.set_facecolor("#bfdbfe")
ax5.set_ylabel("ROC-AUC"); ax5.set_title("CV AUC by Model")

plt.tight_layout()
plt.savefig("models/evaluation_report.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Evaluation plots saved → models/evaluation_report.png")

# ─────────────────────────────────────────────
# 11. SAVE MODEL & METADATA
# ─────────────────────────────────────────────
joblib.dump(best_pipe, "models/delay_model.pkl")
joblib.dump({
    "numeric_features":     NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "roc_auc":              roc_auc,
    "avg_precision":        avg_prec,
    "model_name":           best_name,
}, "models/model_metadata.pkl")

print("✓ Model saved → models/delay_model.pkl")
print("✓ Metadata saved → models/model_metadata.pkl")
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
