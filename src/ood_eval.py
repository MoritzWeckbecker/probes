from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from src.generate_data import extract_activations, get_model_and_tokenizer
from src.probe import plot_roc, recall_at_fpr


def build_datapoints(yaml_path: str):
    items = yaml.safe_load(Path(yaml_path).read_text())
    datapoints = []
    for item_id, item in enumerate(items):
        for completion, label in ((item["honest_completion"], 0), (item["deceptive_completion"], 1)):
            datapoints.append({
                "item_id": item_id,
                "user": f"{item['scenario']}\n\n{item['question']}",
                "assistant": f"{item['answer_prefix']} {completion}",
                "label": label,
            })
    return datapoints


def extract_all_layers(datapoints, model_name: str):
    model, tokenizer = get_model_and_tokenizer(model_name)
    per_layer = {}
    for n, dp in enumerate(datapoints):
        layer_acts = extract_activations(model, tokenizer, dp["user"], dp["assistant"])
        for layer, act in layer_acts.items():
            per_layer.setdefault(layer, []).append(act)
        if (n + 1) % 100 == 0:
            print(f"extracted {n + 1}/{len(datapoints)}", flush=True)

    labels = torch.tensor([dp["label"] for dp in datapoints], dtype=torch.long)
    item_ids = torch.tensor([dp["item_id"] for dp in datapoints], dtype=torch.long)
    return {
        layer: {
            "ACTIVATIONS": torch.stack(acts).float(),
            "LABELS": labels,
            "ITEM_IDS": item_ids,
        }
        for layer, acts in per_layer.items()
    }


def main(
    yaml_path: str = "data/roleplay.yaml",
    probe_path: str = "data/probe_layer14.pt",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    roc_path: str = "data/roc_ood_layer14.png",
    cache_path: str = "data/ood_activations.pt",
):
    probe = torch.load(probe_path)
    layer = probe["layer"]
    datapoints = build_datapoints(yaml_path)
    print(f"{len(datapoints)} OOD datapoints from {len(datapoints) // 2} scenarios")

    data = extract_all_layers(datapoints, model_name)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(f"saved OOD activations for {len(data)} layers to {cache_path}")

    X = data[layer]["ACTIVATIONS"].numpy()
    y_np = data[layer]["LABELS"].numpy()
    z = (X - probe["mean"].numpy()) / probe["std"].numpy()
    scores = z @ probe["weights"].numpy() + probe["bias"]

    fpr, tpr, _ = roc_curve(y_np, scores)
    auroc = roc_auc_score(y_np, scores)
    recall = recall_at_fpr(y_np, scores, 0.01)

    print(f"OOD AUROC (layer {layer}): {auroc:.3f}")
    print(f"OOD recall at 1% FPR (layer {layer}): {recall:.3f}")
    for name, label in (("deceptive (1)", 1), ("honest (0)", 0)):
        v = scores[y_np == label]
        print(f"{name}: n={len(v)} min={v.min():.2f} median={np.median(v):.2f} max={v.max():.2f}")
    print(f"errors at threshold 0: FN={int(((y_np == 1) & (scores < 0)).sum())}, FP={int(((y_np == 0) & (scores >= 0)).sum())}")

    plot_roc(fpr, tpr, auroc, Path(roc_path), layer)
    print(f"saved ROC to {roc_path}")


if __name__ == "__main__":
    main()
