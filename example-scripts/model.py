import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

msgs = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Is the Earth flat?"}]
inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                    return_tensors='pt', return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=100,
                     #do_sample=True, temperature=1.0)
                     do_sample=False)
prompt_len = inputs["input_ids"].shape[1]
print(tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True))