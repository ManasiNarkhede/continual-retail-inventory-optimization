import torch

ckpt = torch.load("checkpoints/tft/tft_latest.pt", map_location="cpu")
print("Type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Keys:", list(ckpt.keys())[:10])
    for k, v in list(ckpt.items())[:5]:
        if hasattr(v, "shape"):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {type(v)}")
else:
    print("Not a dict checkpoint")
