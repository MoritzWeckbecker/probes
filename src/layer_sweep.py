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
    reg_lambda: float = 10.0,
    roc_path: str = "data/roc_ood_all_layers.png",
    results_path: str = "data/layer_sweep_results.pt",
):
    ood = torch.load(ood_path)
    num_layers = len(ood)
    results = {}

    for layer in range(num_layers):
        X, y, groups = load_layer(train_path, layer)
        (Xtr, ytr, _), (Xte, yte, _) = group_split(X, y, groups)
        clf, mean, std = train_probe(Xtr, ytr, reg_lambda)

        test_scores = clf.decision_function((Xte - mean) / std)
        test_auc = roc_auc_score(yte, test_scores)

        X_ood = ood[layer]["ACTIVATIONS"].numpy()
        y_ood = ood[layer]["LABELS"].numpy()
        ood_scores = clf.decision_function((X_ood - mean) / std)
        fpr, tpr, _ = roc_curve(y_ood, ood_scores)

        results[layer] = {
            "weights": torch.tensor(clf.coef_[0], dtype=torch.float32),
            "bias": float(clf.intercept_[0]),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "test_auc": test_auc,
            "ood_auc": roc_auc_score(y_ood, ood_scores),
            "ood_recall_at_1pct_fpr": recall_at_fpr(y_ood, ood_scores, 0.01),
            "ood_fpr": fpr,
            "ood_tpr": tpr,
        }
        print(
            f"layer {layer:2d}: test AUC {test_auc:.3f} | "
            f"OOD AUC {results[layer]['ood_auc']:.3f} | "
            f"OOD recall@1%FPR {results[layer]['ood_recall_at_1pct_fpr']:.3f}",
            flush=True,
        )

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, results_path)
    print(f"saved per-layer probes and metrics to {results_path}")

    cols = 6
    rows = int(np.ceil(num_layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for layer in range(num_layers):
        ax = axes[layer // cols][layer % cols]
        r = results[layer]
        ax.plot(r["ood_fpr"], r["ood_tpr"], color="C0")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
        ax.set_title(f"layer {layer}, AUROC {r['ood_auc']:.3f}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[num_layers:]:
        ax.set_visible(False)
    fig.supxlabel("False positive rate (OOD)")
    fig.supylabel("True positive rate (OOD)")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    print(f"saved ROC grid to {roc_path}")


if __name__ == "__main__":
    main()
