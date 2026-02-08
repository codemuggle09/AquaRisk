import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

def visualize_state_fuzzy_report(state_summary, output_dir="results"):
    """
    Generates visualizations for state-wise fluoride fuzzy safety report.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n📡 Generating FUZZY visualizations...")

    states = state_summary.index
    fluoride_means = state_summary["Fluoride (mg/L)"]
    risk_scores = state_summary["Risk_Score"]

    # Expand Risk_Label counts into proper columns
    risk_expanded = state_summary["Risk_Label"].apply(pd.Series).fillna(0)
    for key in ["Low Risk", "Medium Risk", "High Risk"]:
        if key not in risk_expanded.columns:
            risk_expanded[key] = 0

    # ---------- 1. Mean Fluoride per State ----------
    plt.figure(figsize=(14, 6))
    plt.bar(states, fluoride_means)
    plt.xticks(rotation=90)
    plt.ylabel("Mean Fluoride (mg/L)")
    plt.title("State-wise Mean Fluoride Concentration")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_fluoride_per_state.png")
    plt.close()

    # ---------- 2. Mean Risk Score per State ----------
    plt.figure(figsize=(14, 6))
    plt.bar(states, risk_scores)
    plt.xticks(rotation=90)
    plt.ylabel("Mean Fuzzy Risk Score")
    plt.title("State-wise Mean Fuzzy Risk Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_risk_score_per_state.png")
    plt.close()

    # ---------- 3. Stacked Bar – Risk Labels ----------
    plt.figure(figsize=(14, 7))
    plt.bar(states, risk_expanded["Low Risk"], label="Low Risk")
    plt.bar(states, risk_expanded["Medium Risk"], 
            bottom=risk_expanded["Low Risk"], label="Medium Risk")
    plt.bar(states, risk_expanded["High Risk"],
            bottom=risk_expanded["Low Risk"] + risk_expanded["Medium Risk"],
            label="High Risk")

    plt.xticks(rotation=90)
    plt.ylabel("Count")
    plt.title("Risk Label Distribution per State")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stacked_risk_labels_per_state.png")
    plt.close()

    # ---------- 4. Heatmap (Risk Score) ----------
    plt.figure(figsize=(10, 12))
    plt.imshow(risk_scores.values.reshape(-1, 1), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Risk Score")
    plt.yticks(np.arange(len(states)), labels=states)
    plt.title("Fuzzy Risk Score Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_state_risk_scores.png")
    plt.close()

    print("✅ Fuzzy visualizations generated.")


# =======================================================================================
# 🔥 NEW: MACHINE LEARNING MODEL PERFORMANCE VISUALIZATION
# =======================================================================================

def visualize_classification_results(y_true, y_pred, model_name="Model", output_dir="results"):
    """
    Creates visualizations for confusion matrix and classification report.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n📡 Generating ML Model visualizations...")

    # ---------- 1. Confusion Matrix Heatmap ----------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"{model_name} – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()

    # ---------- 2. Classification Report Bars ----------
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    df_report = df_report.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

    plt.figure(figsize=(10, 6))
    df_report[['precision', 'recall', 'f1-score']].plot(kind='bar')
    plt.title(f"{model_name} – Classification Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_classification_report.png")
    plt.close()

    print(f"✅ Model visualizations saved for {model_name}")

