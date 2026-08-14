import torch
from model import Encoder
from model_poseformer import PoseFormerEncoder

x = torch.randn(4, 1, 75, 117)
for name, M in [('cnn', Encoder), ('poseformer', PoseFormerEncoder)]:
    m = M()
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f'{name}: out={tuple(y.shape)} params={n_params:,}')
    y.sum().backward()
    n_with_grad = sum(
        1 for p in m.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    n_total = sum(1 for _ in m.parameters())
    print(f'{name}: params with non-zero grad: {n_with_grad}/{n_total}')
