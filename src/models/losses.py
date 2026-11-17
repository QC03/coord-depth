"""
Composite Loss Functions for Z-Depth Refinement.
Combines Z-Depth L1/SmoothL1, 3D MPJPE, and Bone Length Consistency regularizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skills.joint_topology.joint_mapping import SKELETON_EDGES


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
    ) -> None:
        super().__init__()
        self.w_depth = weight_depth
        self.w_3d = weight_3d
        self.w_bone = weight_bone

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
        loss_bone = torch.tensor(0.0, device=x_mp.device)
        if self.w_bone > 0:
            bone_diffs = []
            for u, v in self.edges:
                pred_bone = torch.norm(pred_3d[:, u] - pred_3d[:, v], dim=-1)
                gt_bone = torch.norm(gt_3d[:, u] - gt_3d[:, v], dim=-1)
                bone_diffs.append(torch.abs(pred_bone - gt_bone))
            loss_bone = torch.mean(torch.stack(bone_diffs))

        # Total Loss
        total_loss = (
            self.w_depth * loss_depth
            + self.w_3d * loss_3d
            + self.w_bone * loss_bone
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_depth": loss_depth.item(),
            "loss_3d": loss_3d.item(),
            "loss_bone": loss_bone.item(),
        }

        return total_loss, metrics
