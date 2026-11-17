"""
Composite Loss Functions for Z-Depth Refinement.
Combines Z-Depth L1/SmoothL1, 3D MPJPE, and Bone Length Consistency regularizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skills.joint_topology.joint_mapping import SKELETON_EDGES


DEPTH_SEPARATION_EDGES = [(10, 11), (11, 12), (13, 14), (14, 15)]
ANGLE_LIMITS = (
    (0, 1, 2, 120.0),
    (1, 2, 3, 150.0),
    (0, 4, 5, 120.0),
    (4, 5, 6, 150.0),
    (16, 10, 11, 150.0),
    (10, 11, 12, 150.0),
    (16, 13, 14, 150.0),
    (13, 14, 15, 150.0),
    (7, 8, 9, 90.0),
)


def bone_length_loss(
    pred_3d: torch.Tensor,
    gt_3d: torch.Tensor,
    edges: list[tuple[int, int]] = SKELETON_EDGES,
) -> torch.Tensor:
    differences = [
        torch.abs(
            torch.linalg.vector_norm(pred_3d[:, u] - pred_3d[:, v], dim=-1)
            - torch.linalg.vector_norm(gt_3d[:, u] - gt_3d[:, v], dim=-1)
        )
        for u, v in edges
    ]
    return torch.mean(torch.stack(differences))


def depth_separation_loss(
    pred_3d: torch.Tensor,
    gt_3d: torch.Tensor,
    edges: list[tuple[int, int]] = DEPTH_SEPARATION_EDGES,
) -> torch.Tensor:
    """Require the Z gap needed to preserve a bone when its XY projection is short."""
    penalties = []
    for u, v in edges:
        target_length = torch.linalg.vector_norm(gt_3d[:, u] - gt_3d[:, v], dim=-1)
        xy_length = torch.linalg.vector_norm(pred_3d[:, u, :2] - pred_3d[:, v, :2], dim=-1)
        required_dz = torch.sqrt(torch.clamp(target_length.square() - xy_length.square(), min=0.0))
        actual_dz = torch.abs(pred_3d[:, u, 2] - pred_3d[:, v, 2])
        penalties.append(torch.relu(required_dz - actual_dz).square())
    return torch.mean(torch.stack(penalties))


def joint_angle_limit_loss(
    pred_3d: torch.Tensor,
    limits: tuple[tuple[int, int, int, float], ...] = ANGLE_LIMITS,
) -> torch.Tensor:
    """Penalize bends beyond clinical ROM limits observable from three joints."""
    penalties = []
    for parent, joint, child, max_bend_degrees in limits:
        first = pred_3d[:, parent] - pred_3d[:, joint]
        second = pred_3d[:, child] - pred_3d[:, joint]
        cosine = torch.sum(first * second, dim=-1) / (
            torch.linalg.vector_norm(first, dim=-1)
            * torch.linalg.vector_norm(second, dim=-1)
        ).clamp_min(1e-6)
        internal_angle = torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        bend_angle = torch.pi - internal_angle
        max_bend = torch.as_tensor(max_bend_degrees, device=pred_3d.device) * torch.pi / 180.0
        penalties.append(torch.relu(bend_angle - max_bend).square())
    return torch.mean(torch.stack(penalties))


class DepthRefinementLoss(nn.Module):
    """
    Multi-objective loss function:
    Loss = w_depth * L_depth + w_3d * L_3d + w_bone * L_bone
    """

    def __init__(
        self,
        weight_depth: float = 1.0,
        weight_3d: float = 0.5,
        weight_bone: float = 0.2,
        weight_depth_separation: float = 0.1,
        weight_angle: float = 0.05,
    ) -> None:
        super().__init__()
        self.w_depth = weight_depth
        self.w_3d = weight_3d
        self.w_bone = weight_bone
        self.w_depth_separation = weight_depth_separation
        self.w_angle = weight_angle

        self.smooth_l1 = nn.SmoothL1Loss(beta=0.01)
        self.edges = SKELETON_EDGES

    def forward(
        self,
        x_mp: torch.Tensor,
        delta_z: torch.Tensor,
        target_z: torch.Tensor,
        gt_3d: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            x_mp: (B, 17, 3) [X_mp, Y_mp, Z_mp]
            delta_z: (B, 17, 1) predicted residual
            target_z: (B, 17, 1) ground truth Z
            gt_3d: (B, 17, 3) full ground truth pose
        """
        # 1. Refined Z & Pose
        refined_z = x_mp[:, :, 2:3] + delta_z
        pred_3d = torch.cat([x_mp[:, :, :2], refined_z], dim=-1)

        # 2. Depth Loss
        loss_depth = self.smooth_l1(refined_z, target_z)

        # 3. 3D MPJPE Loss
        loss_3d = torch.mean(torch.norm(pred_3d - gt_3d, dim=-1))

        # 4. Bone Length Consistency Loss
        loss_bone = bone_length_loss(pred_3d, gt_3d, self.edges)
        loss_depth_separation = depth_separation_loss(pred_3d, gt_3d)
        loss_angle = joint_angle_limit_loss(pred_3d)

        # Total Loss
        total_loss = (
            self.w_depth * loss_depth
            + self.w_3d * loss_3d
            + self.w_bone * loss_bone
            + self.w_depth_separation * loss_depth_separation
            + self.w_angle * loss_angle
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_depth": loss_depth.item(),
            "loss_3d": loss_3d.item(),
            "loss_bone": loss_bone.item(),
            "loss_depth_separation": loss_depth_separation.item(),
            "loss_angle": loss_angle.item(),
        }

        return total_loss, metrics
