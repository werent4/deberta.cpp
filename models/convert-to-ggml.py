import sys
import struct
import json
import torch
import numpy as np
import os
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

if len(sys.argv) < 2:
    print("Usage: convert-to-ggml.py dir-model [ftype]")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

dir_model = sys.argv[1]
ftype = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ftype_str = ["f32", "f16"]
fname_out = dir_model + f"/ggml-model-{ftype_str[ftype]}.bin"

if not os.path.exists(dir_model):
    model_name = "microsoft/deberta-v3-base"
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.save_pretrained(dir_model)
    model.save_pretrained(dir_model)

with open(dir_model + "/config.json", "r") as f:
    hparams = json.load(f)

print("Config:", json.dumps(hparams, indent=2))

tokenizer = AutoTokenizer.from_pretrained(dir_model)
model = AutoModel.from_pretrained(dir_model, low_cpu_mem_usage=True)

print(model)

list_vars = model.state_dict()
for name, tensor in list_vars.items():
    if "embed_proj" in name:
        print(f"{name}: shape={tensor.shape}")  
    # print(name, tensor.shape, tensor.dtype)

# weights to skip
SKIP = {
    'embeddings.position_ids',
    'mask_predictions.classifier.weight',
    'mask_predictions.classifier.bias',
    'mask_predictions.dense.weight',
    'mask_predictions.dense.bias',
    'mask_predictions.LayerNorm.weight',
    'mask_predictions.LayerNorm.bias',
}

fout = open(fname_out, "wb")

# magic
fout.write(struct.pack("i", 0x67676d6c))

# hparams
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["max_position_embeddings"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["intermediate_size"]))
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
fout.write(struct.pack("i", hparams.get("position_buckets", 256)))
fout.write(struct.pack("i", hparams.get("max_relative_positions", -1)))
fout.write(struct.pack("i", ftype))
fout.write(struct.pack("i", hparams.get("embedding_size", hparams["hidden_size"])))
fout.write(struct.pack("i", hparams.get("type_vocab_size", 0)))
fout.write(struct.pack("i", int(hparams.get("position_biased_input", True))))
fout.write(struct.pack("f", hparams.get("layer_norm_eps", 1e-7)))

# tensors
for name, tensor in list_vars.items():
    # strip "deberta." prefix если есть
    short_name = name.replace("deberta.", "", 1)

    if any(skip in short_name for skip in SKIP):
        print(f"Skipping {name}")
        continue

    data = tensor.squeeze().numpy()
    n_dims = len(data.shape)

    if ftype == 1 and name.endswith(".weight") and n_dims == 2:
        print(f"  Converting {name} to float16")
        data = data.astype(np.float16)
        l_type = 1
    else:
        data = data.astype(np.float32)
        l_type = 0

    if "embed" in name.lower():
        print(f"Writing {name} shape={data.shape} dtype={data.dtype}")

    encoded_name = name.encode("utf-8")
    fout.write(struct.pack("iii", n_dims, len(encoded_name), l_type))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(encoded_name)
    data.tofile(fout)

fout.close()
print(f"\nDone. Output: {fname_out}")