import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, sensitivity (macro recall), specificity (macro), and error rate
    for MULTI-CLASS classification.
    """

    # ----- Accuracy -----
    accuracy = accuracy_score(y_true, y_pred)

    # ----- Sensitivity (Macro Recall) -----
    sensitivity = recall_score(y_true, y_pred, average='macro')

    # ----- Specificity (Macro Specificity) -----
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificity_list = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_list.append(specificity)

    specificity = np.mean(specificity_list)

    # ----- Error rate -----
    error_rate = 1 - accuracy

    return accuracy, sensitivity, specificity, error_rate


def visualize_all_model_metrics(results_dict, output_dir="results"):
    """
    results_dict = {
        "LR": (y_true, y_pred),
        "SVM": (...),
        "RF": (...),
        ...
    }
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = list(results_dict.keys())

    accuracies = []
    sensitivities = []
    specificities = []
    error_rates = []

    # ---- Compute all metrics for each model ----
    for model in models:
        y_true, y_pred = results_dict[model]
        acc, sen, spe, err = compute_metrics(y_true, y_pred)

        accuracies.append(acc)
        sensitivities.append(sen)
        specificities.append(spe)
        error_rates.append(err)

    x = np.arange(len(models))

    # ---- Create 2x2 subplot layout like your image ----
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Accuracy
    axs[0, 0].bar(x, accuracies, color=['green','red','yellow','blue','orange'][:len(models)])
    axs[0, 0].set_title("(a) Accuracy of ML Models")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(models)

    # (b) Sensitivity
    axs[0, 1].bar(x, sensitivities, color=['green','red','yellow','blue','orange'][:len(models)])
    axs[0, 1].set_title("(b) Sensitivity of ML Models")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(models)

    # (c) Specificity
    axs[1, 0].bar(x, specificities, color=['green','red','yellow','blue','orange'][:len(models)])
    axs[1, 0].set_title("(c) Specificity of ML Models")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(models)

    # (d) Error Rate
    axs[1, 1].bar(x, error_rates, color=['green','red','yellow','blue','orange'][:len(models)])
    axs[1, 1].set_title("(d) Error Rate of ML Models")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(models)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ML_model_comparison_subplots.png")
    plt.close()

    # ----------- OVERALL PERFORMANCE PLOT (e) -----------
    plt.figure(figsize=(12, 6))
    width = 0.2

    plt.bar(x - width*1.5, accuracies, width, label="Accuracy")
    plt.bar(x - width*0.5, sensitivities, width, label="Sensitivity")
    plt.bar(x + width*0.5, specificities, width, label="Specificity")
    plt.bar(x + width*1.5, error_rates, width, label="Error Rate")

    plt.xticks(x, models)
    plt.title("(e) Overall Performance of ML Models")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ML_model_overall_performance.png")
    plt.close()

    print("✅ ML Model performance visualizations saved in /results folder")
