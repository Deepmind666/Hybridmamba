import torch

print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(2)

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print(f"mean={float(y.mean()):.6f}")
