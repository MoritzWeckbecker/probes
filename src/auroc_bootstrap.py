from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.probe import recall_at_fpr


def bar_with_ci(layer_ids, points, lows, highs, ylabel, title, out_path, chance=None, ylim=None):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(layer_ids, points, color="C0", alpha=0.8)
    ax.errorbar(
        layer_ids, points,
        yerr=[points - lows, highs - points],
        fmt="none", ecolor="black", capsize=3, linewidth=1,
    )
    if chance is not None:
        ax.axhline(chance, linestyle="--", color="gray", linewidth=1, label="chance")
        ax.legend(loc="upper left")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layer_ids)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(
    sweep_path: str = "data/layer_sweep_results.pt",
    ood_path: str = "data/ood_activations.pt",
    B: int = 1000,
    seed: int = 0,
    out_prefix: str = "data/",
    suffix: str = "7b",
):
    sweep = torch.load(sweep_path, weights_only=False)
    ood = torch.load(ood_path)
    rng = np.random.default_rng(seed)

    item_ids = ood[0]["ITEM_IDS"].numpy()
    y = ood[0]["LABELS"].numpy()
    items = np.unique(item_ids)
    item_rows = [np.where(item_ids == i)[0] for i in items]

    layers = sorted(sweep)
    auroc_rows, recall_rows, accuracy_rows = [], [], []
    for layer in layers:
        p = sweep[layer]
        X = ood[layer]["ACTIVATIONS"].numpy()
        z = (X - p["mean"].numpy()) / p["std"].numpy()
        scores = z @ p["weights"].numpy() + p["bias"]
        correct = ((scores >= 0).astype(int) == y).astype(float)

        auroc_point = roc_auc_score(y, scores)
        recall_point = recall_at_fpr(y, scores, 0.01)
        accuracy_point = correct.mean()
        auroc_boot = np.empty(B)
        recall_boot = np.empty(B)
        accuracy_boot = np.empty(B)
        for b in range(B):
            sampled = rng.integers(0, len(items), len(items))
            idx = np.concatenate([item_rows[j] for j in sampled])
            auroc_boot[b] = roc_auc_score(y[idx], scores[idx])
            recall_boot[b] = recall_at_fpr(y[idx], scores[idx], 0.01)
            accuracy_boot[b] = correct[idx].mean()

        a_lo, a_hi = np.percentile(auroc_boot, [2.5, 97.5])
        r_lo, r_hi = np.percentile(recall_boot, [2.5, 97.5])
        c_lo, c_hi = np.percentile(accuracy_boot, [2.5, 97.5])
        auroc_rows.append((layer, auroc_point, a_lo, a_hi))
        recall_rows.append((layer, recall_point, r_lo, r_hi))
        accuracy_rows.append((layer, accuracy_point, c_lo, c_hi))
        print(
            f"layer {layer:2d}: AUROC {auroc_point:.3f} [{a_lo:.3f}, {a_hi:.3f}] | "
            f"recall@1%FPR {recall_point:.3f} [{r_lo:.3f}, {r_hi:.3f}] | "
            f"accuracy {accuracy_point:.3f} [{c_lo:.3f}, {c_hi:.3f}]",
            flush=True,
        )

    layer_ids = [r[0] for r in auroc_rows]
    for name, rows, ylabel, chance, ylim in [
        ("auroc", auroc_rows, "OOD AUROC", 0.5, (0, 1)),
        ("recall", recall_rows, "OOD recall at 1% FPR", None, None),
        ("accuracy", accuracy_rows, "OOD accuracy (threshold 0)", 0.5, (0, 1)),
    ]:
        points = np.array([r[1] for r in rows])
        lows = np.array([r[2] for r in rows])
        highs = np.array([r[3] for r in rows])
        csv_path = f"{out_prefix}{name}_ci_{suffix}.csv"
        with open(csv_path, "w") as f:
            f.write("layer,value,ci_low,ci_high\n")
            for layer, point, lo, hi in rows:
                f.write(f"{layer},{point:.4f},{lo:.4f},{hi:.4f}\n")
        bar_with_ci(
            layer_ids, points, lows, highs, ylabel,
            f"Qwen2.5-{suffix.upper()}-Instruct, {ylabel} per layer (95% bootstrap CI, B={B})",
            f"{out_prefix}{name}_per_layer_{suffix}.png",
            chance=chance, ylim=ylim,
        )
        print(f"saved {name} plot and CIs to {csv_path}")


if __name__ == "__main__":
    main()
