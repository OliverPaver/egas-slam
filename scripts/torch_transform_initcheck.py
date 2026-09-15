import torch
v = torch.randn(256, 3, device="cuda")
h = torch.ones(v.shape[0], 4, device="cuda")
h[:, :3] = v
m = torch.eye(4, device="cuda")
a = (m @ h.T).T
b = a[:, :3] / a[:, 3:4]
torch.cuda.synchronize()
print("maximum error", (b-v).abs().max().item(), flush=True)
