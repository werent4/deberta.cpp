import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
cpp_binary = os.path.join(ROOT_DIR, "build", "examples", "example-batch-cpu")
model_bin  = os.path.join(ROOT_DIR, "ggml-deberta", "ggml-model-f32.bin")
cpp_out    = os.path.join(ROOT_DIR, "cpp_batch_out.txt")
model_name = "microsoft/deberta-v3-base"

text_batch = [
    "The cat sat on the mat.",
    "Deep learning models require large amounts of training data to generalize well.",
]

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(text_batch, return_tensors="pt", padding=True)

input_ids = inputs["input_ids"].tolist()
attention_mask = inputs["attention_mask"].tolist()
seq_len    = len(input_ids[0])
batch_size = len(input_ids)

expected_ids = [
    [279, 3185, 3208, 277, 262, 8358, 260, 0, 0, 0, 0, 0, 0],
    [7222, 1101, 1836, 1449, 614, 3909, 265, 838, 514, 264, 57304, 371, 260]
]
expected_mask = [
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
assert input_ids == expected_ids, f"input_ids mismatch!\ngot:      {input_ids}\nexpected: {expected_ids}"
assert attention_mask == expected_mask, f"attention_mask mismatch!\ngot:      {attention_mask}\nexpected: {expected_mask}"
print("assertions passed\n")

print(f"batch_size={batch_size}, seq_len={seq_len}")
print(f"input_ids[0]:      {input_ids[0]}")
print(f"input_ids[1]:      {input_ids[1]}")
print(f"attention_mask[0]: {attention_mask[0]}")
print(f"attention_mask[1]: {attention_mask[1]}\n")

# --- run C++ binary ---
# тут надо будет адаптировать под то как твой example-batch-cpu принимает инпут
result = subprocess.run([cpp_binary, model_bin], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("stderr:", result.stderr)
    sys.exit(1)

# --- load cpp output ---
cpp = []
with open(cpp_out) as f:
    for line in f:
        cpp.append([float(x) for x in line.split()])
cpp = np.array(cpp)  # (batch_size * seq_len, 768)
cpp = cpp.reshape(batch_size, seq_len, 768)

# --- python forward ---
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
pt = outputs.last_hidden_state.cpu().numpy()  # (batch_size, seq_len, 768)

# --- compare per batch item ---
for b in range(batch_size):
    real_len = int(sum(attention_mask[b]))
    cpp_b = cpp[b, :real_len]
    pt_b  = pt[b, :real_len]

    diff = np.abs(cpp_b - pt_b)
    print(f"=== batch {b} (real tokens: {real_len}) ===")
    print(f"max abs diff:    {diff.max():.6f}")
    print(f"mean abs diff:   {diff.mean():.6f}")
    print(f"median abs diff: {np.median(diff):.6f}")

    cos = F.cosine_similarity(torch.tensor(cpp_b), torch.tensor(pt_b), dim=1)
    print(f"mean cos sim:    {cos.mean():.6f}")
    print(f"min  cos sim:    {cos.min():.6f}  at token={cos.argmin().item()}\n")