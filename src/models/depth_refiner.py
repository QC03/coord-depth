"""
Z-Depth Refinement Neural Networks.
Pure XY preservation with residual Z prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear layer with LayerNorm, ReLU, and Residual Skip connection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ZResidualMLP(nn.Module):
    """
    Predicts depth residual Delta Z for each joint.
    Input:  (B, 17, 3) [X_mp, Y_mp, Z_mp]
    Output: (B, 17, 1) [Delta_Z]
    """

    def __init__(
        self,
        num_joints: int = 17,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints

        input_dim = num_joints * 3
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_joints),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, 17, 3) float32
        Returns:
            delta_z: (B, 17, 1) float32
        """
        B = x.shape[0]
        flat_x = x.reshape(B, -1)

        feat = self.input_proj(flat_x)
        for block in self.blocks:
            feat = block(feat)

        delta_z = self.output_head(feat) # (B, 17)
        return delta_z.reshape(B, self.num_joints, 1)

    @torch.no_grad()
    def predict_3d_pose(self, x_mp: torch.Tensor) -> torch.Tensor:
        """
        Guarantees XY coordinates are 100% frozen from MediaPipe,
        combining them with the refined Z prediction.
        Returns: (B, 17, 3) [X_mp, Y_mp, Z_refined]
        """
        delta_z = self.forward(x_mp)
        refined_z = x_mp[:, :, 2:3] + delta_z
        # Strict XY invariant: concatenate original XY with refined Z
        return torch.cat([x_mp[:, :, :2], refined_z], dim=-1)
