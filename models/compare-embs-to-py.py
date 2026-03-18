import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# --- paths ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))  # ~/deberta.cpp/models
ROOT_DIR   = os.path.dirname(BASE_DIR) # ~/deberta.cpp

cpp_binary = os.path.join(ROOT_DIR, "build", "main")
model_bin  = os.path.join(ROOT_DIR, "ggml-deberta", "ggml-model-f32.bin")
cpp_out    = os.path.join(ROOT_DIR, "cpp_out.txt")
model_name = "microsoft/deberta-v3-base"

DEFAULT_TEXT = (
"Born in Funchal, Madeira, Ronaldo began his career with Sporting CP before signing with Manchester United in 2003."
"He became a star player at United, where he won three consecutive Premier League titles, the Champions League, and the FIFA Club World Cup."
"His 2007–08 season earned him his first Ballon d'Or at age 23. In 2009, Ronaldo became the subject of the then-most expensive"
"transfer in history when he joined Real Madrid in a deal worth €94 million (£80 million)."
"At Madrid, he was at the forefront of the club's resurgence as a dominant European force, "
"helping them win four Champions Leagues between 2014 and 2018, including the long-awaited La Décima."
"He also won two La Liga titles, including the record-breaking 2011–12 season in which Madrid reached 100 points, and became the club's all-time top goalscorer."
"He won Ballon d'Ors in 2013, 2014, 2016 and 2017, and was runner-up three times to Lionel Messi, his perceived career rival."
"Following issues with the club hierarchy, Ronaldo signed for Juventus in 2018 in a transfer worth an initial €100 million,"
"where he was pivotal in winning two Serie A titles. In 2021, he returned to United before joining Al-Nassr in 2023."
)

text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
print(f"text: {text!r}\n")

# --- tokenize ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"][0].tolist()
print(f"tokens ({len(input_ids)})\n")

# --- run C++ binary ---
cmd = [cpp_binary, model_bin] + [str(t) for t in input_ids]
print(f"running: {cmd[:4]}...\n")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("stderr:", result.stderr)
    sys.exit(1)

# --- load cpp output ---
cpp = []
with open(cpp_out) as f:
    for line in f:
        cpp.append([float(x) for x in line.split()])
cpp = np.array(cpp)  # (seq_len, 768)

# --- python forward ---
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

print("first 10 values of python output:")
print(outputs.last_hidden_state[0, 0, :10])

pt = outputs.last_hidden_state[0].cpu().numpy()  # (seq_len, 768)

# --- compare ---
assert cpp.shape == pt.shape, f"shape mismatch: cpp={cpp.shape} pt={pt.shape}"
diff = np.abs(cpp - pt)
print(f"\nmax abs diff:     {diff.max():.6f}")
print(f"mean abs diff:    {diff.mean():.6f}")
print(f"median abs diff:  {np.median(diff):.6f}")
print(f"99th percentile:  {np.percentile(diff, 99):.6f}")

cpp_t = torch.tensor(cpp)
pt_t  = torch.tensor(pt)
cos   = F.cosine_similarity(cpp_t, pt_t, dim=1)
print(f"\nmean cos sim: {cos.mean():.6f}")
print(f"min  cos sim: {cos.min():.6f}  at token={cos.argmin().item()}")