import os
from typing import Dict, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy_float32(value: ArrayLike) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32).cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def save_latent_golden(
    latent: ArrayLike,
    goldens_dir: str,
    filename: str = "latent_goldens.npy",
) -> str:
    os.makedirs(goldens_dir, exist_ok=True)
    path = os.path.join(goldens_dir, filename)
    np.save(path, _to_numpy_float32(latent))
    return path


def save_audio_golden(
    audio: ArrayLike,
    goldens_dir: str,
    filename: str = "audio_goldens.npy",
) -> str:
    os.makedirs(goldens_dir, exist_ok=True)
    path = os.path.join(goldens_dir, filename)
    np.save(path, _to_numpy_float32(audio))
    return path


def verify_golden_match(
    current: ArrayLike,
    golden_path: str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Union[bool, Tuple[int, ...], float]]:
    current_np = _to_numpy_float32(current)
    golden_np = np.load(golden_path)

    same_shape = current_np.shape == golden_np.shape
    if not same_shape:
        return {
            "match": False,
            "same_shape": False,
            "current_shape": current_np.shape,
            "golden_shape": golden_np.shape,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
        }

    diff = np.abs(current_np - golden_np)
    max_abs_diff = float(np.max(diff)) if diff.size > 0 else 0.0

    denom = np.maximum(np.abs(golden_np), np.finfo(np.float32).eps)
    rel_diff = diff / denom
    max_rel_diff = float(np.max(rel_diff)) if rel_diff.size > 0 else 0.0

    match = bool(np.allclose(current_np, golden_np, rtol=rtol, atol=atol))
    return {
        "match": match,
        "same_shape": True,
        "current_shape": current_np.shape,
        "golden_shape": golden_np.shape,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
    }
