from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("ggml-deberta")
model = AutoModel.from_pretrained("ggml-deberta", dtype=torch.float32)  # todo: select dtype based on model.wtype
model.eval()

input_ids = torch.tensor([[1, 31414, 232, 328, 2]])
with torch.no_grad():
    out = model(input_ids)

print(out.last_hidden_state[0, 0, :8])  