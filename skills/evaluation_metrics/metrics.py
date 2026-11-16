"""
Evaluation Metrics for 3D Pose and Z-Depth Refinement.
Calculates Z-MAE, MPJPE, PA-MPJPE, and PCK in millimeters (mm).
"""

from typing import Union, Tuple
import numpy as np
import torch


def compute_z_mae(
    pred_z: Union[np.ndarray, torch.Tensor],
    gt_z: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes Z-Axis Mean Absolute Error in millimeters (mm).
    Inputs are assumed to be in meters.
    """
    if isinstance(pred_z, torch.Tensor):
        diff = torch.abs(pred_z - gt_z)
        mae_m = torch.mean(diff).item()
    else:
        diff = np.abs(pred_z - gt_z)
        mae_m = float(np.mean(diff))
    return mae_m * 1000.0  # meter -> mm


def compute_mpjpe(
    pred_3d: Union[np.ndarray, torch.Tensor],
    gt_3d: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes Mean Per Joint Position Error (MPJPE) in millimeters (mm).
    Inputs: (N, J, 3) in meters.
    """
    if isinstance(pred_3d, torch.Tensor):
        dist = torch.norm(pred_3d - gt_3d, dim=-1) # (N, J)
        mpjpe_m = torch.mean(dist).item()
    else:
        dist = np.linalg.norm(pred_3d - gt_3d, axis=-1)
        mpjpe_m = float(np.mean(dist))
    return mpjpe_m * 1000.0  # meter -> mm


def compute_pck(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
    threshold_mm: float = 150.0
) -> float:
    """
    Computes Percentage of Correct Keypoints (%) within threshold_mm.
    """
    dist_mm = np.linalg.norm(pred_3d - gt_3d, axis=-1) * 1000.0
    correct = dist_mm < threshold_mm
    return float(np.mean(correct) * 100.0)


def compute_pa_mpjpe(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray
) -> float:
    """
    Computes Procrustes Aligned MPJPE (PA-MPJPE) in mm.
    Applies optimal rotation and translation to align pred with gt.
    Inputs: (N, J, 3) in meters.
    """
    if pred_3d.ndim == 2:
        pred_3d = pred_3d[np.newaxis, ...]
        gt_3d = gt_3d[np.newaxis, ...]

    aligned_preds = []
    for p, g in zip(pred_3d, gt_3d):
        # Center both
        mu_p = p.mean(axis=0, keepdims=True)
        mu_g = g.mean(axis=0, keepdims=True)
        p_c = p - mu_p
        g_c = g - mu_g

        # SVD for optimal rotation
        H = p_c.T @ g_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        scale = np.sum(S) / np.sum(p_c ** 2) if np.sum(p_c ** 2) > 1e-8 else 1.0
        p_aligned = scale * (p_c @ R.T) + mu_g
        aligned_preds.append(p_aligned)

    aligned_preds = np.array(aligned_preds)
    return compute_mpjpe(aligned_preds, gt_3d)
