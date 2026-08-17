"""Batched forward kinematics for the body-only joint chain shared across
SMPL, SMPL-H, and SMPL-X: joint 0 (pelvis) + joints 1-21 (``body_pose``).
These 22 joints have identical indices/semantics in all three model
flavors -- that's exactly where the three kinematic trees diverge (hands
and face get appended after index 21) -- so this module never needs to
know which of the three produced a given take.

Deliberately bypasses ``smplx.body_models.SMPLX``/``SMPLH``/``SMPL``: the
PSU100 release ships pruned model pkls (no hand PCA basis, no landmark
data), which those classes fail to load, and none of that is needed
anyway since the body joint chain depends only on ``J_regressor``,
``kintree_table``, ``shapedirs``, and ``v_template``. Reuses ``smplx.lbs``'s
own numeric primitives (rest-pose joint regression + rigid kinematic
chain), so the result is identical to what the full model would produce
for these 22 joints -- just without paying for vertex skinning.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

BODY_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
NUM_BODY_JOINTS = len(BODY_JOINT_NAMES)  # 22 (pelvis + 21 body_pose joints)

# MoSh++/AMASS-style npz stores this as `surface_model_type`; maps to the
# body_models/ subdirectory and file prefix.
MODEL_TYPE_SUBDIRS = {"smplx": "smplx", "smplh": "smplh_p", "smpl": "smpl"}


@dataclass
class BodyModel:
    v_template: torch.Tensor    # (V, 3)
    shapedirs: torch.Tensor     # (V, 3, num_betas)
    J_regressor: torch.Tensor   # (NUM_BODY_JOINTS, V)
    parents: torch.Tensor       # (NUM_BODY_JOINTS,), parents[0] == -1


def load_body_model(
    body_models_dir: Path, model_type: str, gender: str,
    num_betas: int = 16, device: str = "cpu",
) -> BodyModel:
    subdir = MODEL_TYPE_SUBDIRS.get(model_type, model_type)
    pkl_path = Path(body_models_dir) / subdir / f"{model_type.upper()}_{gender.upper()}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    v_template = torch.tensor(np.array(data["v_template"]), dtype=torch.float32, device=device)
    shapedirs = torch.tensor(
        np.array(data["shapedirs"])[:, :, :num_betas], dtype=torch.float32, device=device
    )
    J_regressor = torch.tensor(
        np.array(data["J_regressor"])[:NUM_BODY_JOINTS], dtype=torch.float32, device=device
    )
    parents = torch.tensor(
        np.array(data["kintree_table"])[0][:NUM_BODY_JOINTS].astype(np.int64), device=device
    )
    return BodyModel(v_template, shapedirs, J_regressor, parents)


def compute_body_joints(
    body_model: BodyModel,
    betas: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    device: str = "cpu",
    chunk_size: int = 4096,
) -> np.ndarray:
    """Pose the 22-joint body chain for every frame of a take.

    Parameters
    ----------
    betas : (num_betas,) -- constant shape for the whole take.
    global_orient : (T, 3) axis-angle root rotation.
    body_pose : (T, 63) axis-angle, joints 1-21 (SMPL/SMPL-H/SMPL-X's shared
        ``body_pose`` parameter).
    transl : (T, 3) global translation.

    Returns
    -------
    (T, 22, 3) float32 joint positions, pelvis first.
    """
    betas_t = torch.tensor(np.asarray(betas), dtype=torch.float32, device=device).unsqueeze(0)
    v_shaped = body_model.v_template.unsqueeze(0) + blend_shapes(betas_t, body_model.shapedirs)
    J_rest = vertices2joints(body_model.J_regressor, v_shaped)  # (1, 22, 3); shape-dependent, pose-independent

    global_orient_t = torch.as_tensor(global_orient, dtype=torch.float32, device=device)
    body_pose_t = torch.as_tensor(body_pose, dtype=torch.float32, device=device)
    transl_t = torch.as_tensor(transl, dtype=torch.float32, device=device)
    T = global_orient_t.shape[0]

    out = np.empty((T, NUM_BODY_JOINTS, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            n = end - start
            pose = torch.cat(
                [
                    global_orient_t[start:end].unsqueeze(1),
                    body_pose_t[start:end].reshape(n, NUM_BODY_JOINTS - 1, 3),
                ],
                dim=1,
            ).reshape(-1, 3)
            rot_mats = batch_rodrigues(pose).view(n, NUM_BODY_JOINTS, 3, 3)
            J_batch = J_rest.expand(n, -1, -1)
            J_transformed, _ = batch_rigid_transform(
                rot_mats, J_batch, body_model.parents, dtype=torch.float32
            )
            joints = J_transformed + transl_t[start:end].unsqueeze(1)
            out[start:end] = joints.cpu().numpy()
    return out


def compute_skeleton_scale(body_model: BodyModel, betas: np.ndarray) -> float:
    """A single scalar summarizing this body shape's size: mean rest-pose
    (T-pose) distance from the pelvis to the other 21 joints. Pose-
    independent (depends only on betas), so it's a stable per-take
    reference -- unlike the per-window std normalization in
    extract_smplx_patches.py's _normalize_patch, which is re-estimated
    from just the window's 75 frames and so carries some sampling noise,
    and implicitly entangles "how big is this body" with "how much did it
    move in this particular window."

    This is a way of exploiting one thing that's genuinely SMPL-specific:
    with markers there's no ground-truth skeleton to measure from, but the
    forward-kinematics rest pose gives us exact bone lengths for free.
    """
    betas_t = torch.tensor(np.asarray(betas), dtype=torch.float32).unsqueeze(0)
    v_shaped = body_model.v_template.cpu().unsqueeze(0) + blend_shapes(betas_t, body_model.shapedirs.cpu())
    J_rest = vertices2joints(body_model.J_regressor.cpu(), v_shaped)[0]  # (22, 3)
    dists = torch.linalg.norm(J_rest - J_rest[0:1], dim=-1)[1:]  # drop pelvis-to-itself
    return float(dists.mean())


def load_take_joints(
    npz_path: Path, body_models_dir: Path,
    device: str = "cpu", num_betas: int = 16, chunk_size: int = 4096,
    body_model: Optional[BodyModel] = None,
    betas_override: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int]:
    """Load a MoSh++-style ``*_stageii.npz`` and pose its 22-joint body chain.

    Pass a pre-loaded ``body_model`` (see ``load_body_model``) to skip
    re-reading the body-model pkl for every take -- useful when batch
    processing many takes that share a (model_type, gender).

    Pass ``betas_override`` to re-pose the take's real motion
    (``root_orient``/``pose_body``/``trans``) on a *different* body shape
    -- e.g. a jittered version of the take's own betas for shape
    augmentation (see generate_smplx_patch_dataset.py's --beta-augment-k).
    Leaves the (real) betas stored in the npz untouched.

    Returns ``(joints (T, 22, 3) float32, fps)``.
    """
    npz = np.load(npz_path, allow_pickle=True)
    model_type = str(npz["surface_model_type"])
    gender = str(npz["gender"])
    fps = int(npz["mocap_frame_rate"])

    if body_model is None:
        body_model = load_body_model(body_models_dir, model_type, gender, num_betas=num_betas, device=device)
    betas = betas_override if betas_override is not None else np.asarray(npz["betas"])[:num_betas]
    joints = compute_body_joints(
        body_model,
        betas=betas,
        global_orient=npz["root_orient"],
        body_pose=npz["pose_body"],
        transl=npz["trans"],
        device=device,
        chunk_size=chunk_size,
    )
    return joints, fps
