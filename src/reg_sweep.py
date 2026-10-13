from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.probe import group_split, load_layer, recall_at_fpr, train_probe


def main(
    train_path: str = "data/activations.pt",
    ood_path: str = "data/ood_activations.pt",
    layer: int = 14,
    lambdas=(100.0, 10.0, 1.0, 0.1),
    plot_path: str = "data/roc_reg_sweep_layer14.png",
):
    X, y, groups = load_layer(train_path, layer)
    (Xtr, ytr, _), _ = group_split(X, y, groups)

    ood = torch.load(ood_path)
    X_ood = ood[layer]["ACTIVATIONS"].numpy()
    y_ood = ood[layer]["LABELS"].numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    for reg_lambda in lambdas:
        clf, mean, std = train_probe(Xtr, ytr, reg_lambda)
        scores = clf.decision_function((X_ood - mean) / std)
        fpr, tpr, _ = roc_curve(y_ood, scores)
        auroc = roc_auc_score(y_ood, scores)
        recall = recall_at_fpr(y_ood, scores, 0.01)
        print(f"lambda {reg_lambda:>6}: OOD AUROC {auroc:.3f} | recall@1%FPR {recall:.3f} | "
              f"weight norm {np.linalg.norm(clf.coef_):.3f}", flush=True)
        ax.plot(fpr, tpr, label=f"λ={reg_lambda:g} (AUROC {auroc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.axvline(0.01, linestyle=":", color="red", linewidth=0.8, label="1% FPR")
    ax.set_xlabel("False positive rate (OOD)")
    ax.set_ylabel("True positive rate (OOD)")
    ax.set_title(f"Layer {layer} OOD ROC across L2 regularisation")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"saved ROC plot to {plot_path}")


if __name__ == "__main__":
    main()
