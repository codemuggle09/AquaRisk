import os
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_accuracies(acc_dict):
    plt.figure(figsize=(7, 4))
    names = list(acc_dict.keys())
    values = list(acc_dict.values())
    plt.bar(names, values, color="skyblue")
    plt.title("Model Accuracies")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_accuracies.png")
    plt.savefig(path)
    plt.close()
    return path
