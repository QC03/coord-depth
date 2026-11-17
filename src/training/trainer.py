"""
Trainer for MediaPipe XY + Neural Z-Depth Refinement.
Supports mixed precision, validation Z-MAE tracking, and best model checkpointing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from skills.evaluation_metrics.metrics import compute_z_mae, compute_mpjpe


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        checkpoint_dir: str | Path = "checkpoints",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler

        self.best_val_mae = float("inf")
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for x_mp, target_z, gt_3d in pbar:
            x_mp = x_mp.to(self.device)
            target_z = target_z.to(self.device)
            gt_3d = gt_3d.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            delta_z = self.model(x_mp)
            loss, metrics = self.criterion(x_mp, delta_z, target_z, gt_3d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"train_loss": total_loss / max(1, num_batches)}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        pred_z_list = []
        gt_z_list = []
        pred_3d_list = []
        gt_3d_list = []

        for x_mp, target_z, gt_3d in self.val_loader:
            x_mp = x_mp.to(self.device)
            target_z = target_z.to(self.device)
            gt_3d = gt_3d.to(self.device)

            delta_z = self.model(x_mp)
            loss, _ = self.criterion(x_mp, delta_z, target_z, gt_3d)
            total_loss += loss.item()

            refined_z = x_mp[:, :, 2:3] + delta_z
            pred_3d = torch.cat([x_mp[:, :, :2], refined_z], dim=-1)

            pred_z_list.append(refined_z.cpu())
            gt_z_list.append(target_z.cpu())
            pred_3d_list.append(pred_3d.cpu())
            gt_3d_list.append(gt_3d.cpu())

        all_pred_z = torch.cat(pred_z_list, dim=0)
        all_gt_z = torch.cat(gt_z_list, dim=0)
        all_pred_3d = torch.cat(pred_3d_list, dim=0)
        all_gt_3d = torch.cat(gt_3d_list, dim=0)

        val_z_mae = compute_z_mae(all_pred_z, all_gt_z)
        val_mpjpe = compute_mpjpe(all_pred_3d, all_gt_3d)
        num_batches = len(self.val_loader)

        return {
            "val_loss": total_loss / max(1, num_batches),
            "val_z_mae_mm": val_z_mae,
            "val_mpjpe_mm": val_mpjpe,
        }

    def fit(self, num_epochs: int = 20) -> Path:
        print(f"🚀 Starting training on {self.device} for {num_epochs} epochs...")
        best_ckpt_path = self.checkpoint_dir / "best_model.pth"

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            log_entry = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(log_entry)

            val_mae = val_metrics["val_z_mae_mm"]
            is_best = val_mae < self.best_val_mae
            status = " [BEST ⭐]" if is_best else ""

            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Z-MAE: {val_mae:.2f} mm | "
                f"Val MPJPE: {val_metrics['val_mpjpe_mm']:.2f} mm{status}"
            )

            if is_best:
                self.best_val_mae = val_mae
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_mae": self.best_val_mae,
                    },
                    best_ckpt_path,
                )

        # Save history
        history_path = self.checkpoint_dir / "train_history.json"
        history_path.write_text(json.dumps(self.history, indent=2))
        print(f"✅ Training completed! Best Val Z-MAE: {self.best_val_mae:.2f} mm. Saved to {best_ckpt_path}")
        return best_ckpt_path
