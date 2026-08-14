"""Quick sanity check: every registered encoder runs forward + backward
cleanly and produces gradients for all its parameters.

    python -m models._smoke_test
"""

import torch

from . import MODEL_REGISTRY, build_model

x = torch.randn(4, 1, 75, 117)
for name in MODEL_REGISTRY:
    m = build_model(name)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{name}: out={tuple(y.shape)} params={n_params:,}")
    y.sum().backward()
    n_with_grad = sum(
        1 for p in m.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    n_total = sum(1 for _ in m.parameters())
    print(f"{name}: params with non-zero grad: {n_with_grad}/{n_total}")
