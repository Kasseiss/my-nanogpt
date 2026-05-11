import os
import pickle
import torch
from model import GPTConfig, GPT

# ====================== 固定配置，直接运行 ======================
out_dir = 'out'
start = "Hello"
num_samples = 1
max_new_tokens = 300
temperature = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
model = GPT(GPTConfig(**ckpt['model_args']))
model.load_state_dict(ckpt['model'])
model.eval()
model.to(device)

# 加载词表
with open(os.path.join('data', 'mytext', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 生成
x = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]
with torch.no_grad():
    y = model.generate(x, max_new_tokens, temperature=temperature)
    print("\n=== 生成的文本 ===")
    print(decode(y[0].tolist()))
    print("="*50 + "\n")