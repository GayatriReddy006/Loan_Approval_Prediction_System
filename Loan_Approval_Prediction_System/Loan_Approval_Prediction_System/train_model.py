"""Train and compare three loan approval classifiers."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
STATIC_DIR = ROOT / "static"
TARGET_COLUMN = "Loan_Status"  # canonical target name used in this project
IDENTIFIER_CANDIDATES = {"loan_id", "loanid"}
RANDOM_STATE = 42


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, str, str | None]:
    """Read CSV and normalize header whitespace/case without altering values."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the provided loan dataset at data/loan_data.csv."
        )
    data = pd.read_csv(path)
    data.columns = [str(column).strip() for column in data.columns]
    # Accept either the requested classic schema or the supplied lowercase schema.
    target = next((c for c in data.columns if c.casefold() == "loan_status"), None)
    if target is None:
        raise ValueError("CSV must contain a Loan_Status (or loan_status) target column.")
    identifier = next((c for c in data.columns if c.casefold() in IDENTIFIER_CANDIDATES), None)
    if target != TARGET_COLUMN:
        data = data.rename(columns={target: TARGET_COLUMN})
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()
    return data, TARGET_COLUMN, identifier


def make_pipeline(numeric_columns: list[str], categorical_columns: list[str], model,
                  scale_numeric: bool) -> Pipeline:
    """Keep imputation, encoding, scaling and estimator together for safe reuse."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", Pipeline(categorical_steps), categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", model)])


def export_tree(tree) -> dict:
    """Convert a fitted sklearn decision tree into data the browser can evaluate."""
    # Callers may pass either the estimator or its sklearn Tree structure.
    structure = tree if hasattr(tree, "children_left") else tree.tree_
    return {
        "left": structure.children_left.tolist(),
        "right": structure.children_right.tolist(),
        "feature": structure.feature.tolist(),
        "threshold": structure.threshold.tolist(),
        "values": structure.value[:, 0, :].tolist(),
    }


def export_browser_model(name: str, pipeline: Pipeline) -> dict:
    """Export the fitted preprocessing and classifier for local JavaScript inference."""
    preprocessing = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    numeric_transformer = preprocessing.named_transformers_["numeric"]
    categorical_transformer = preprocessing.named_transformers_["categorical"]
    numeric_features = list(preprocessing.transformers_[0][2])
    categorical_features = list(preprocessing.transformers_[1][2])
    encoder = categorical_transformer.named_steps["onehot"]

    numeric_specs = {}
    for index, feature in enumerate(numeric_features):
        spec = {"median": float(numeric_transformer.named_steps["imputer"].statistics_[index])}
        if "scaler" in numeric_transformer.named_steps:
            scaler = numeric_transformer.named_steps["scaler"]
            spec["mean"] = float(scaler.mean_[index])
            spec["scale"] = float(scaler.scale_[index])
        numeric_specs[feature] = spec

    categorical_specs = {}
    for index, feature in enumerate(categorical_features):
        categorical_specs[feature] = {
            "most_frequent": str(categorical_transformer.named_steps["imputer"].statistics_[index]),
            "categories": [str(value) for value in encoder.categories_[index]],
        }

    shared = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_specs": numeric_specs,
        "categorical_specs": categorical_specs,
        "classes": [str(value) for value in classifier.classes_],
    }
    if name == "logistic_regression":
        shared["kind"] = "logistic_regression"
        shared["coefficients"] = classifier.coef_[0].tolist()
        shared["intercept"] = float(classifier.intercept_[0])
    elif name == "decision_tree":
        shared["kind"] = "decision_tree"
        shared["tree"] = export_tree(classifier.tree_)
    else:
        shared["kind"] = "random_forest"
        shared["trees"] = [export_tree(estimator.tree_) for estimator in classifier.estimators_]
    return shared


def main() -> None:
    from sklearn.metrics import roc_auc_score

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data, target_column, identifier = load_dataset()

    # Show a transparent data audit before training.
    print(f"Dataset shape: {data.shape}")
    print("Columns:", data.columns.tolist())
    print("Data types:\n", data.dtypes)
    print("Missing values:\n", data.isna().sum())
    if identifier:
        print(f"Excluding identifier from model input: {identifier}")

    X = data.drop(columns=[target_column] + ([identifier] if identifier else []))
    y = data[target_column].astype(str).str.strip()
    # Treat the approved outcome as the common positive class for all metrics.
    labels = sorted(y.unique().tolist())
    approved_label = next((label for label in labels if label.casefold() == "approved"), None)
    if approved_label is None:
        raise ValueError(f"Expected an Approved class in {target_column}; found {labels}")
    if len(labels) != 2:
        raise ValueError(f"Expected a binary target; found {len(labels)} classes: {labels}")

    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "logistic_regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True),
        "decision_tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
        # Keep the estimator single-process for reliable execution in classroom environments.
        "random_forest": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), False),
    }
    rows = []
    matrix_rows = []
    browser_models = {}
    rf_pipeline = None

    approved_idx = labels.index(approved_label)

    for name, (estimator, scale_numeric) in models.items():
        pipeline = make_pipeline(numeric_columns, categorical_columns, estimator, scale_numeric)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        approved_probs = probabilities[:, approved_idx]

        matrix = confusion_matrix(y_test, predictions, labels=labels)
        auc = float(roc_auc_score(y_test == approved_label, approved_probs))

        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_approved": float(precision_score(y_test, predictions, pos_label=approved_label,
                                                        zero_division=0)),
            "recall_approved": float(recall_score(y_test, predictions, pos_label=approved_label,
                                                  zero_division=0)),
            "f1_approved": float(f1_score(y_test, predictions, pos_label=approved_label,
                                          zero_division=0)),
            "roc_auc": auc,
        })
        matrix_rows.append((name, matrix))
        browser_models[name] = export_browser_model(name, pipeline)
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
        print(f"Saved models/{name}.joblib")
        if name == "random_forest":
            rf_pipeline = pipeline

    # Extract feature importances from the Random Forest model
    rf_clf = rf_pipeline.named_steps["classifier"]
    rf_prep = rf_pipeline.named_steps["preprocessing"]
    encoded_feature_names = rf_prep.get_feature_names_out()
    feature_importances = [
        {"feature": name.replace("numeric__", "").replace("categorical__", ""),
         "importance": round(float(imp), 4)}
        for name, imp in zip(encoded_feature_names, rf_clf.feature_importances_)
    ]
    feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    with (RESULTS_DIR / "evaluation_results.json").open("w", encoding="utf-8") as file:
        json.dump({"positive_class": approved_label, "class_order": labels,
                   "test_rows": len(y_test), "metrics": rows,
                   "confusion_matrices": {name: matrix.tolist() for name, matrix in matrix_rows},
                   "feature_importances": feature_importances},
                  file, indent=2)

    # 1. Compare metrics plot with modern styling
    metric_columns = ["accuracy", "precision_approved", "recall_approved", "f1_approved", "roc_auc"]
    display_names = {
        "accuracy": "Accuracy",
        "precision_approved": "Precision (Approved)",
        "recall_approved": "Recall (Approved)",
        "f1_approved": "F1-Score",
        "roc_auc": "ROC-AUC"
    }
    plot_df = metrics.set_index("model")[metric_columns].rename(columns=display_names)
    plot_df.index = [name.replace("_", " ").title() for name in plot_df.index]

    plt.figure(figsize=(10, 5.5))
    ax = plot_df.plot(kind="bar", figsize=(10, 5.5), colormap="viridis", width=0.75)
    ax.set_title("Loan Approval Classification: Model Performance Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Algorithm", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    plt.xticks(rotation=0, ha="center", fontsize=10)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=180)
    plt.close()

    # 2. Confusion Matrices plot
    fig, axes = plt.subplots(1, len(matrix_rows), figsize=(14, 4.2))
    for axis, (name, matrix) in zip(axes, matrix_rows):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 11, "weight": "bold"})
        axis.set_title(name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axis.set_xlabel("Predicted Label", fontsize=10)
        axis.set_ylabel("Actual Label", fontsize=10)
    fig.suptitle("Confusion Matrices on Held-Out Test Set (N = 854)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    # 3. Feature Importance plot
    plt.figure(figsize=(9, 5))
    top_features = pd.DataFrame(feature_importances).head(8).sort_values(by="importance", ascending=True)
    colors = ["#10b981" if "cibil" in f else "#6366f1" for f in top_features["feature"]]
    bars = plt.barh(top_features["feature"], top_features["importance"], color=colors, height=0.6)
    plt.title("Random Forest: Top Feature Importances (Gini Importance)", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Relative Importance Weight", fontsize=10)
    plt.xlim(0, max(top_features["importance"]) * 1.15)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=180)
    plt.close()

    # GUI choices and typical numeric values come from the real training data.
    metadata = {
        "features": list(X.columns),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "categorical_values": {c: sorted(X[c].dropna().astype(str).unique().tolist())
                               for c in categorical_columns},
        "numeric_defaults": {c: float(X[c].median()) for c in numeric_columns},
        "positive_class": approved_label,
        "classes": labels,
        "identifier_excluded": identifier,
        "feature_importances": feature_importances,
    }
    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    # This JavaScript data file lets index.html run directly in a browser without Flask.
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    browser_data = {"metadata": metadata, "models": browser_models,
                    "metrics": rows, "class_order": labels,
                    "feature_importances": feature_importances}
    with (STATIC_DIR / "model_data.js").open("w", encoding="utf-8") as file:
        file.write("window.LOAN_MODEL_DATA = ")
        json.dump(browser_data, file, separators=(",", ":"))
        file.write(";\n")

    print("\nEvaluation results (Approved is positive):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nResults saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
