from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit


def load_layer(data_path: str, layer: int):
    data = torch.load(data_path)[layer]
    X = data["ACTIVATIONS"].numpy()
    y = data["LABELS"].numpy()
    groups = data["FACT_IDS"].numpy()
    return X, y, groups


def group_split(X, y, groups, test_size: float = 0.2, seed: int = 0):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        (X[train_idx], y[train_idx], groups[train_idx]),
        (X[test_idx], y[test_idx], groups[test_idx]),
    )


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return tpr[fpr <= max_fpr].max()


def train_probe(X_train, y_train, reg_lambda: float = 10.0):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X_train - mean) / std

    clf = LogisticRegression(C=1.0 / reg_lambda, max_iter=5000)
    clf.fit(X_norm, y_train)
    return clf, mean, std


def evaluate_probe(clf, mean, std, X_test, y_test):
    scores = clf.decision_function((X_test - mean) / std)
    fpr, tpr, _ = roc_curve(y_test, scores)
    return {
        "auc": roc_auc_score(y_test, scores),
        "recall_at_1pct_fpr": recall_at_fpr(y_test, scores, 0.01),
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_roc(fpr, tpr, auc, out_path: Path, layer: int):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.axvline(0.01, linestyle=":", color="red", label="1% FPR")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Probe ROC, layer {layer} (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(
    data_path: str = "data/activations.pt",
    layer: int = 14,
    reg_lambda: float = 10.0,
    probe_path: str = "data/probe_layer14.pt",
    roc_path: str = "data/roc_layer14.png",
):
    X, y, groups = load_layer(data_path, layer)
    (X_train, y_train, groups_train), (X_test, y_test, groups_test) = group_split(X, y, groups)
    print(f"train: {len(y_train)} datapoints, test: {len(y_test)} datapoints")
    print(f"train/test fact overlap: {len(set(groups_train) & set(groups_test))}")

    clf, mean, std = train_probe(X_train, y_train, reg_lambda)
    train_acc = clf.score((X_train - mean) / std, y_train)

    metrics = evaluate_probe(clf, mean, std, X_test, y_test)
    print(f"train accuracy: {train_acc:.3f}")
    print(f"test AUC: {metrics['auc']:.3f}")
    print(f"test recall at 1% FPR: {metrics['recall_at_1pct_fpr']:.3f}")

    probe_path = Path(probe_path)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer": layer,
            "weights": torch.tensor(clf.coef_[0], dtype=torch.float32),
            "bias": float(clf.intercept_[0]),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "reg_lambda": reg_lambda,
        },
        probe_path,
    )
    print(f"saved probe to {probe_path}")

    plot_roc(metrics["fpr"], metrics["tpr"], metrics["auc"], Path(roc_path), layer)
    print(f"saved ROC to {roc_path}")


if __name__ == "__main__":
    main()
