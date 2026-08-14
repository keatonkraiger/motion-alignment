"""Model registry: build a Siamese encoder by name.

All four encoders share the same input/output API -- input
``(N, 1, num_frame, num_joints * in_chans)``, output ``(N, out_dim)`` --
so they're interchangeable inside the Siamese/DTW pipeline. ``cnn`` takes
no constructor arguments; the other three accept architecture params
(``embed_dim_ratio``, ``depth``, ``num_heads``, ...), which is what a
config's ``model.params`` maps onto.
"""

from __future__ import annotations

import torch.nn as nn

from .cnn import CNNEncoder
from .mixste import MixSTEEncoder
from .poseformer import PoseFormerEncoder
from .poseformerv2 import PoseFormerV2Encoder

MODEL_REGISTRY = {
    "cnn": CNNEncoder,
    "poseformer": PoseFormerEncoder,
    "mixste": MixSTEEncoder,
    "poseformerv2": PoseFormerV2Encoder,
}

# Per-encoder optimizer presets pulled from each method's own original
# training recipe. Used when a config's training.use_original_hparams is
# true, in place of the CNN baseline's SGD recipe.
#
# lr_div_batchsize mirrors the CNN baseline's ``lr / batch_size`` quirk;
# the transformer methods' original code uses the LR as-is.
ORIGINAL_HPARAMS = {
    "cnn": {
        "optimizer": "SGD",
        "lr": 0.01,
        "weight_decay": 5e-4,
        "lr_decay": 1.0,
        "lr_div_batchsize": True,
    },
    "poseformer": {
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.1,
        "lr_decay": 0.99,
        "lr_div_batchsize": False,
    },
    "poseformerv2": {
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.1,
        "lr_decay": 0.99,
        "lr_div_batchsize": False,
    },
    "mixste": {
        "optimizer": "AdamW",
        "lr": 4e-5,
        "weight_decay": 0.1,
        "lr_decay": 0.99,
        "lr_div_batchsize": False,
    },
}


def build_model(name: str, params: dict | None = None) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name!r}. Expected one of {sorted(MODEL_REGISTRY)}."
        )
    return MODEL_REGISTRY[name](**(params or {}))
