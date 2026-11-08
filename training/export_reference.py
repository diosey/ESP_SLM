# export_reference.py
import torch
from model import BitNetTransformer

CHECKPOINT = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\checkpoints_1M\ep20_val1.620.pt"
TOKEN_IDS = [4, 10, 20, 30, 40]

print("=" * 50)
print("Reference logits (PyTorch)")
print("=" * 50)

ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = BitNetTransformer(**ckpt["config"])
model.load_state_dict(ckpt["model_state"])
model.eval()

with torch.no_grad():
    idx = torch.tensor([TOKEN_IDS])
    logits = model(idx)[0]  # [T, vocab_size]

for pos in range(len(TOKEN_IDS)):
    top = torch.topk(logits[pos], 5)
    print(f"pos {pos} (token {TOKEN_IDS[pos]}):")
    for k in range(5):
        print(f"  #{k+1}: id={int(top.indices[k])}, logit={float(top.values[k]):+.6f}")
