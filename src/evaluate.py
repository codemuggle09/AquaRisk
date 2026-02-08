"""
Evaluate saved models and visualize class-wise metrics & confusion matrices.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_models():
    models = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            name = f.replace("_model.pkl", "")
            models[name] = joblib.load(os.path.join(MODEL_DIR, f))
    return models


def compute_specificity(cm):
    spec = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        spec.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
    return spec


def evaluate_models(models, X_test, y_test):
    results_list = []

    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        cm = confusion_matrix(y_test, preds)
        recall = recall_score(y_test, preds, average=None)   # Sensitivity
        spec = compute_specificity(cm)
        err = 1 - recall  # Error rate per class

        # Save confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix (Acc={acc:.2f})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{name}_cm.png"))
        plt.close()

        print(f"\n🧠 Model: {name}")
        print(f"✅ Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        results_list.append({
            "Model": name,
            "Accuracy": acc,
            "Sensitivity_0": recall[0], "Sensitivity_1": recall[1], "Sensitivity_2": recall[2],
            "Specificity_0": spec[0], "Specificity_1": spec[1], "Specificity_2": spec[2],
            "Error_0": err[0], "Error_1": err[1], "Error_2": err[2]
        })

    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(RESULTS_DIR, "evaluation_summary_full.csv"), index=False)
    print("\n📁 Saved results to: results/evaluation_summary_full.csv")

    plot_metrics(df)


def plot_metrics(df):
    models = df["Model"]
    metrics = ["Accuracy", "Sensitivity", "Specificity", "Error"]

    for metric in ["Sensitivity", "Specificity", "Error"]:
        plt.figure(figsize=(8, 4))
        for c in range(3):
            plt.bar(models, df[f"{metric}_{c}"], label=f"Class {c}", alpha=0.7)
        plt.title(f"{metric} by Class")
        plt.legend()
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric.lower()}_comparison.png"))
        plt.close()

    # Overall summary
    plt.figure(figsize=(8,4))
    plt.bar(models, df["Accuracy"], label="Accuracy")
    plt.title("Overall Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_summary.png"))
    plt.close()
