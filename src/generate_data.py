import csv
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_facts(csv_path: str) -> List[str]:
    """Load fact statements from CSV file."""
    facts = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row['Thstatement'])  # Note: typo in CSV header
    return facts


NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."

PLAIN_USERS = [
    "Pretend you're an honest person making statements about the world.",
    "Pretend you're a dishonest person making statements about the world.",
]

# variant -> (user message per label [honest, deceptive], system prefix per label; None uses NEUTRAL_SYSTEM_PROMPT)
VARIANTS = {
    "plain": (PLAIN_USERS, [None, None]),
    "plain_with_sys": (PLAIN_USERS, ["You are an AI assistant.", "You are an AI assistant."]),
    "you_are": (
        ["You are an honest AI assistant.", "You are a deceptive AI assistant."],
        [None, None],
    ),
    "you_are_fact": (
        ["You are an honest AI assistant. Please tell me a fact.", "You are a deceptive AI assistant. Please tell me a fact."],
        [None, None],
    ),
    "you_are_fact_sys": (
        ["Please tell me a fact.", "Please tell me a fact."],
        ["You are an honest AI assistant.", "You are a deceptive AI assistant."],
    ),
}


def get_model_and_tokenizer(model_name: str):
    """Load model and tokenizer, spread across the GPUs visible to this process."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    return model, tokenizer


@torch.no_grad()
def extract_activations(
    model,
    tokenizer,
    user_prompt: str,
    fact: str,
    ignore_last_n: int = 5,
    system_prompt: str = NEUTRAL_SYSTEM_PROMPT
) -> Dict[int, np.ndarray]:
    """
    Extract activations from all layers for a user prompt and fact pair.
    Uses chat template formatting with user/assistant roles.
    Extracts and averages activations over the assistant (fact) tokens,
    ignoring the last N tokens.

    Returns dict mapping layer index to activations (1D array).
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": fact}
    ]

    full_inputs = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=False,
        return_tensors='pt',
        return_dict=True
    ).to(model.device)

    fact_len = len(tokenizer(fact, add_special_tokens=False)['input_ids'])
    # Qwen's template appends "<|im_end|>\n" after the assistant content
    template_suffix_len = 2

    outputs = model(**full_inputs, output_hidden_states=True)

    # hidden_states is a tuple of tensors, one per layer (including embedding layer)
    # Each tensor is [batch_size, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states

    activations = {}

    for layer_idx, hidden_state in enumerate(hidden_states):
        h = hidden_state.float()  # [1, seq_len, hidden_dim]

        fact_end = -template_suffix_len
        h_assistant = h[:, fact_end - fact_len:fact_end - ignore_last_n, :]

        act = h_assistant.mean(dim=1).squeeze(0).cpu()  # [hidden_dim]
        activations[layer_idx] = act

    return activations


def _worker(rank: int, gpu_ids: List[int], items: List[Tuple], model_name: str, shard_path: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    model, tokenizer = get_model_and_tokenizer(model_name)
    results = {}
    start = time.time()
    for n, (k, fact, prompt, system) in enumerate(items):
        results[k] = extract_activations(model, tokenizer, prompt, fact, system_prompt=system)
        if (n + 1) % 25 == 0:
            print(f"[worker {rank}] {n + 1}/{len(items)} ({time.time() - start:.0f}s)", flush=True)
    torch.save(results, shard_path)


def generate_data(
    csv_path: str = "data/true_false_facts.csv",
    output_path: str = "data/activations.pt",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    gpus_per_worker: int = 1,
    variants: List[str] = list(VARIANTS)
):
    """
    Main function to generate training data, sharded across all visible GPUs.

    Saves as a dictionary with structure:
    {
        layer: {
            'ACTIVATIONS': tensor of shape [num_datapoints, activation_dim],
            'LABELS': tensor of shape [num_datapoints] with 0/1 for honest/deceptive,
            'FACT_IDS': tensor of shape [num_datapoints] with the CSV row index of the fact,
            'VARIANT_IDS': tensor of shape [num_datapoints] with the index into `variants`
        }
    }
    """
    facts = load_facts(csv_path)
    num_gpus = torch.cuda.device_count()
    num_workers = num_gpus // gpus_per_worker
    print(f"Loaded {len(facts)} facts, variants {variants}, {num_workers} workers on {gpus_per_worker} GPU(s) each", flush=True)

    meta = []  # (fact_idx, label, variant_idx) per datapoint k
    items = []  # (k, fact, user_prompt, system_prompt) per datapoint k
    for fact_idx, fact in enumerate(facts):
        for variant_idx, name in enumerate(variants):
            users, sys_prefixes = VARIANTS[name]
            for label in (0, 1):
                k = len(items)
                meta.append((fact_idx, label, variant_idx))
                system = sys_prefixes[label] or NEUTRAL_SYSTEM_PROMPT
                items.append((k, fact, users[label], system))

    shards = [items[r::num_workers] for r in range(num_workers)]
    ctx = mp.get_context("spawn")

    start = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        shard_paths = [f"{tmp_dir}/shard_{r}.pt" for r in range(num_workers)]
        gpu_groups = [
            list(range(r * gpus_per_worker, (r + 1) * gpus_per_worker)) for r in range(num_workers)
        ]
        procs = [
            ctx.Process(target=_worker, args=(r, gpu_groups[r], shards[r], model_name, shard_paths[r]))
            for r in range(num_workers)
        ]
        for p in procs:
            p.start()
        for r, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker {r} failed with exit code {p.exitcode}")

        results = {}
        for path in shard_paths:
            results.update(torch.load(path))
    print(f"Extraction took {(time.time() - start) / 60:.1f} min", flush=True)

    num_datapoints = len(items)
    num_layers = len(results[0])
    labels = torch.tensor([label for _, label, _ in meta], dtype=torch.long)
    fact_ids = torch.tensor([fact_idx for fact_idx, _, _ in meta], dtype=torch.long)
    variant_ids = torch.tensor([variant_idx for _, _, variant_idx in meta], dtype=torch.long)

    output_dict = {}
    for layer in range(num_layers):
        acts = torch.stack([results[k][layer] for k in range(num_datapoints)])
        output_dict[layer] = {
            'ACTIVATIONS': acts.float(),
            'LABELS': labels,
            'FACT_IDS': fact_ids,
            'VARIANT_IDS': variant_ids,
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_dict, output_path)

    print(f"Saved data to {output_path}")
    print(f"Generated {num_datapoints} datapoints")
    print(f"Label distribution: {np.bincount(labels.numpy())}")


if __name__ == "__main__":
    generate_data()
