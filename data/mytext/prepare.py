import os
import numpy as np
import pickle

# 加载文本
with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

# 构建词表
chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# 保存 meta.pkl（最重要！）
meta = {
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

# 生成训练数据
def encode(s):
    return [stoi[c] for c in s]

train_ids = encode(data)
train_ids = np.array(train_ids, dtype=np.uint16)

# 保存 bin 文件
train_ids[:int(len(train_ids)*0.9)].tofile('train.bin')
train_ids[int(len(train_ids)*0.9):].tofile('val.bin')

print("✅ 全部生成完成！")
print(f"词表大小: {vocab_size}")