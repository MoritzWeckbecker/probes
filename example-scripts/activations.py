import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Is the Earth flat?"}]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def get_acts(texts, layer, bs=16):
    feats = []
    for i in range(0, len(texts), bs):
        batch = tokenizer(texts[i:i+bs], return_tensors="pt", padding=True).to(model.device)
        out = model(**batch, output_hidden_states=True)
        h = out.hidden_states[layer].float()               # [batch, seq, d_model]
        m = batch.attention_mask.unsqueeze(-1)     # ignore padding tokens
        feats.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.concatenate(feats)

print(model.config.num_hidden_layers)
acts=get_acts([text], 10, bs=1)
print(acts)