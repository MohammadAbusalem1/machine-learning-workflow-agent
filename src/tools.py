from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATASET_ALIASES = {
    "iris": "iris",
    "iris dataset": "iris",
    "penguin": "penguins",
    "penguins": "penguins",
    "penguin dataset": "penguins",
    "penguins dataset": "penguins",
}

MODEL_ALIASES = {
    "logistic regression": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "logisticregression": "logistic_regression",
    "decision tree": "decision_tree",
    "decision_tree": "decision_tree",
    "decisiontree": "decision_tree",
    "decision tree classifier": "decision_tree",
    "decision_tree_classifier": "decision_tree",
    "knn": "knn",
    "k-nearest neighbors": "knn",
    "k nearest neighbors": "knn",
    "kneighborsclassifier": "knn",
}


@dataclass
class WorkflowState:
    dataset_name: str | None = None
    raw_data: pd.DataFrame | None = None
    X_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_train: pd.Series | np.ndarray | None = None
    y_test: pd.Series | np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    model_name: str | None = None
    model: Any = None
    scaler: StandardScaler | None = None


class MLWorkflowTools:
    """Stateful tool collection used by the agent and by direct Python workflows."""

    def __init__(self, output_dir: str | Path = "artifacts") -> None:
        self.state = WorkflowState()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_dataset(name: str) -> str:
        key = name.lower().strip()
        if key not in DATASET_ALIASES:
            raise ValueError("Supported datasets: iris, penguins.")
        return DATASET_ALIASES[key]

    @staticmethod
    def _normalize_model(name: str) -> str:
        key = name.lower().strip()
        if key not in MODEL_ALIASES:
            raise ValueError(
                "Supported models: logistic regression, decision tree, knn."
            )
        return MODEL_ALIASES[key]

    def model_memory(self, query: str) -> str:
        key = query.lower().strip()
        memory = {
            "iris": "Iris is a 3-class classification dataset with 150 samples and four numeric flower measurements.",
            "iris dataset": "Iris is a 3-class classification dataset with 150 samples and four numeric flower measurements.",
            "penguins": "Palmer Penguins is a 3-species classification dataset with physical measurements and some missing values.",
            "penguins dataset": "Palmer Penguins is a 3-species classification dataset with physical measurements and some missing values.",
            "logistic regression": "Logistic regression is a linear probabilistic classifier. This project uses solver='lbfgs'.",
            "decision tree": "The decision-tree classifier uses max_depth=3 for a compact, interpretable model.",
            "knn": "K-nearest neighbors classifies a sample using the majority class among its five nearest neighbors.",
        }
        return memory.get(
            key,
            "No exact memory entry. Try iris, penguins, logistic regression, decision tree, or knn.",
        )

    def dataset_loader(self, dataset_name: str) -> str:
        dataset = self._normalize_dataset(dataset_name)

        if dataset == "iris":
            bundle = load_iris(as_frame=True)
            df = bundle.frame.rename(columns={"target": "species_id"})
            df["species"] = df["species_id"].map(
                {i: name for i, name in enumerate(bundle.target_names)}
            )
        else:
            # seaborn downloads this small public dataset on first use if it is not cached.
            df = sns.load_dataset("penguins")

        self.state.dataset_name = dataset
        self.state.raw_data = df.copy()
        return (
            f"Loaded {dataset} dataset with {len(df)} rows and {len(df.columns)} columns.\n"
            f"Preview:\n{df.head().to_string(index=False)}"
        )

    def dataset_preprocessing(self, dataset_name: str) -> str:
        dataset = self._normalize_dataset(dataset_name)
        if self.state.raw_data is None or self.state.dataset_name != dataset:
            self.dataset_loader(dataset)

        assert self.state.raw_data is not None
        df = self.state.raw_data.copy()

        if dataset == "iris":
            feature_names = [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ]
            X = df[feature_names]
            y = df["species"]
        else:
            # Keep the classification target as species. The original coursework
            # notebook accidentally used body_mass_g as the target after dropping
            # categorical columns; this refactor corrects that mismatch.
            feature_names = [
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ]
            clean = df[feature_names + ["species"]].dropna().copy()
            X = clean[feature_names]
            y = clean["species"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.state.X_train = X_train_scaled
        self.state.X_test = X_test_scaled
        self.state.y_train = y_train.reset_index(drop=True)
        self.state.y_test = y_test.reset_index(drop=True)
        self.state.feature_names = feature_names
        self.state.class_names = sorted(pd.Series(y).astype(str).unique().tolist())
        self.state.scaler = scaler
        self.state.model = None
        self.state.model_name = None

        return (
            f"Preprocessed {dataset}: {len(X_train)} training rows, "
            f"{len(X_test)} test rows, {len(feature_names)} standardized numeric features."
        )

    def train_model(self, model_name: str) -> str:
        if self.state.X_train is None or self.state.y_train is None:
            raise RuntimeError("Preprocess a dataset before training a model.")

        model_key = self._normalize_model(model_name)
        if model_key == "logistic_regression":
            model = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
        elif model_key == "decision_tree":
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
        else:
            model = KNeighborsClassifier(n_neighbors=5)

        model.fit(self.state.X_train, self.state.y_train)
        self.state.model_name = model_key
        self.state.model = model
        return f"Trained {model_key.replace('_', ' ')} on {self.state.dataset_name}."

    def evaluate_model(self, _: str = "") -> str:
        if self.state.model is None or self.state.X_test is None or self.state.y_test is None:
            raise RuntimeError("Train a model before evaluation.")

        y_pred = self.state.model.predict(self.state.X_test)
        y_true = self.state.y_test
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        return "Evaluation results:\n" + "\n".join(
            f"- {name}: {value:.4f}" for name, value in metrics.items()
        )

    def visualize_results(self, _: str = "") -> str:
        if self.state.model is None or self.state.X_test is None or self.state.y_test is None:
            raise RuntimeError("Train a model before creating visualizations.")

        model_label = self.state.model_name or "model"
        dataset = self.state.dataset_name or "dataset"
        y_true = np.asarray(self.state.y_test)
        y_pred = np.asarray(self.state.model.predict(self.state.X_test))
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

        # Figure 1: confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix: {model_label.replace('_', ' ').title()} on {dataset.title()}")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        fig.tight_layout()
        cm_path = self.output_dir / f"{dataset}_{model_label}_confusion_matrix.png"
        fig.savefig(cm_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        # Figure 2: class-wise accuracy
        class_accuracy = []
        for label in labels:
            mask = y_true == label
            class_accuracy.append(float(np.mean(y_pred[mask] == y_true[mask])))

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.bar([str(x) for x in labels], class_accuracy)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Class")
        ax.set_title(f"Class-wise Accuracy: {model_label.replace('_', ' ').title()}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        acc_path = self.output_dir / f"{dataset}_{model_label}_class_accuracy.png"
        fig.savefig(acc_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        paths = [cm_path, acc_path]

        # Optional third plot for models with interpretable coefficients/importances.
        importances = None
        if hasattr(self.state.model, "coef_"):
            importances = np.abs(self.state.model.coef_).mean(axis=0)
        elif hasattr(self.state.model, "feature_importances_"):
            importances = self.state.model.feature_importances_

        if importances is not None:
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            order = np.argsort(importances)
            ax.barh(
                [self.state.feature_names[i] for i in order],
                np.asarray(importances)[order],
            )
            ax.set_xlabel("Relative importance")
            ax.set_title(f"Feature Importance: {model_label.replace('_', ' ').title()}")
            fig.tight_layout()
            imp_path = self.output_dir / f"{dataset}_{model_label}_feature_importance.png"
            fig.savefig(imp_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            paths.append(imp_path)

        return "Generated visualizations:\n" + "\n".join(f"- {p}" for p in paths)

    def tool_registry(self) -> dict[str, Any]:
        return {
            "model_memory": self.model_memory,
            "dataset_loader": self.dataset_loader,
            "dataset_preprocessing": self.dataset_preprocessing,
            "train_model": self.train_model,
            "evaluate_model": self.evaluate_model,
            "visualize_results": self.visualize_results,
        }
