import numpy as np
from sklearn.metrics import accuracy_score

from src.probe import group_split, load_layer, train_probe


def balanced_random_labels(n: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(n, dtype=int)
    labels[rng.permutation(n)[: n // 2]] = 1
    return labels


def main(
    train_path: str = "data/activations.pt",
    layer: int = 14,
    lambdas=(100.0, 10.0, 1.0, 0.1),
    seeds=(0, 1, 2, 3, 4),
):
    X, y_true, groups = load_layer(train_path, layer)
    print(f"layer {layer}, {len(y_true)} datapoints, {len(seeds)} shuffle seeds")
    print("| lambda | train acc (mean ± sd) | test acc (mean ± sd) |")
    print("|---|---|---|")
    for reg_lambda in lambdas:
        train_accs, test_accs = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            y_shuf = balanced_random_labels(len(y_true), rng)
            (Xtr, ytr, _), (Xte, yte, _) = group_split(X, y_shuf, groups)
            clf, mean, std = train_probe(Xtr, ytr, reg_lambda)
            train_accs.append(accuracy_score(ytr, clf.predict((Xtr - mean) / std)))
            test_accs.append(accuracy_score(yte, clf.predict((Xte - mean) / std)))
        print(
            f"| {reg_lambda:g} | {np.mean(train_accs):.3f} ± {np.std(train_accs):.3f} | "
            f"{np.mean(test_accs):.3f} ± {np.std(test_accs):.3f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
