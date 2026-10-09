"""Train and compare churn models, then save metrics, figures and the best model.

Run from the project root:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data import load_data, split_features_target

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
SEED = 42

# Chart styling
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 10,
        "figure.dpi": 150,
    }
)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical ones."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def build_models(X: pd.DataFrame, y: pd.Series) -> dict[str, Pipeline]:
    """Three models, each set up to handle the ~73/27 class imbalance."""
    imbalance = (y == 0).sum() / (y == 1).sum()
    estimators = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance,
            eval_metric="logloss",
            random_state=SEED,
        ),
    }
    return {
        name: Pipeline([("prep", build_preprocessor(X)), ("model", est)])
        for name, est in estimators.items()
    }


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    rates = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(rates.index, rates.values, color=COLORS[0], width=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color=INK)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by contract type")
    ax.set_ylim(0, rates.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_by_contract.png")
    plt.close(fig)


def plot_roc_curves(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (name, (fpr, tpr, auc)), color in zip(curves.items(), COLORS):
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png")
    plt.close(fig)


def plot_confusion_matrix(cm, name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
    labels = ["Stayed", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(length=0, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.6
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, color="white" if dark else INK)
    ax.set_title(f"Confusion matrix: {name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png")
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, name: str) -> None:
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(top.index, top.values, color=COLORS[0], height=0.55)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Top 10 features: {name}")
    ax.tick_params(axis="y", colors=INK, length=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    plot_churn_by_contract(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, curves, fitted = {}, {}, {}

    for name, pipe in build_models(X_train, y_train).items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        test_auc = roc_auc_score(y_test, proba)
        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "test_roc_auc": round(float(test_auc), 4),
            "test_precision": round(float(precision_score(y_test, pred)), 4),
            "test_recall": round(float(recall_score(y_test, pred)), 4),
            "test_f1": round(float(f1_score(y_test, pred)), 4),
        }
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves[name] = (fpr, tpr, test_auc)
        fitted[name] = pipe

    # Pick the best model on cross-validation, not on the test set,
    # so the test score stays an honest estimate.
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    best = fitted[best_name]

    plot_roc_curves(curves)
    best_pred = best.predict(X_test)
    plot_confusion_matrix(confusion_matrix(y_test, best_pred), best_name)

    perm = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED
    )
    importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(
        ascending=False
    )
    plot_feature_importance(importance, best_name)

    joblib.dump(best, MODEL_DIR / "best_model.joblib")
    output = {
        "best_model": best_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate": round(float(y.mean()), 4),
        "models": results,
        "top_features": {k: round(float(v), 4) for k, v in importance.head(10).items()},
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(output, indent=2))

    print(pd.DataFrame(results).T.to_string())
    print(f"\nBest model (by cross-validated ROC-AUC): {best_name}")
    print("\nTop features:")
    print(importance.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
