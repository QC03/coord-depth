"""
Main Training & Evaluation Entrypoint for MediaPipe XY + Neural Z-Depth Refinement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.models.depth_refiner import ZResidualMLP
from src.models.losses import DepthRefinementLoss
from src.training.trainer import Trainer
from skills.evaluation_metrics.metrics import compute_z_mae, compute_mpjpe, compute_pck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--stride", type=int, default=3, help="Frame sample stride (higher=faster)")
    parser.add_argument("--max-actions", type=int, default=None, help="Max actions per subject (for quick tests)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"🔧 Device: {device} | Epochs: {args.epochs} | Stride: {args.stride}")

    # 1. Dataloaders
    print("📦 Loading datasets according to subject split...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=ROOT / "train",
        batch_size=args.batch_size,
        stride=args.stride,
        max_actions_per_subj=args.max_actions,
    )
    print(
        f"   Train samples: {len(train_loader.dataset):,} | "
        f"Val samples: {len(val_loader.dataset):,} | "
        f"Test samples: {len(test_loader.dataset):,}"
    )

    # 2. Model & Optimizer
    model = ZResidualMLP(num_joints=17, hidden_dim=args.hidden_dim, num_blocks=3)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = DepthRefinementLoss(weight_depth=1.0, weight_3d=0.5, weight_bone=0.2)

    # 3. Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=ROOT / "checkpoints",
        scheduler=scheduler,
    )
    best_ckpt = trainer.fit(num_epochs=args.epochs)

    # 4. Final Evaluation on Independent Test Subject (s06)
    print("\n🧪 Running final benchmark on Test Set (Subject s06)...")
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    baseline_z_list = []
    refined_z_list = []
    gt_z_list = []
    pred_3d_list = []
    gt_3d_list = []

    with torch.no_grad():
        for x_mp, target_z, gt_3d in test_loader:
            x_mp = x_mp.to(device)
            target_z = target_z.to(device)
            gt_3d = gt_3d.to(device)

            delta_z = model(x_mp)
            refined_z = x_mp[:, :, 2:3] + delta_z
            pred_3d = torch.cat([x_mp[:, :, :2], refined_z], dim=-1)

            baseline_z_list.append(x_mp[:, :, 2:3].cpu())
            refined_z_list.append(refined_z.cpu())
            gt_z_list.append(target_z.cpu())
            pred_3d_list.append(pred_3d.cpu())
            gt_3d_list.append(gt_3d.cpu())

    all_base_z = torch.cat(baseline_z_list, dim=0)
    all_ref_z = torch.cat(refined_z_list, dim=0)
    all_gt_z = torch.cat(gt_z_list, dim=0)
    all_pred_3d = torch.cat(pred_3d_list, dim=0)
    all_gt_3d = torch.cat(gt_3d_list, dim=0)

    base_z_mae = compute_z_mae(all_base_z, all_gt_z)
    ref_z_mae = compute_z_mae(all_ref_z, all_gt_z)
    improvement = ((base_z_mae - ref_z_mae) / base_z_mae) * 100.0

    print("=" * 60)
    print("📊 FINAL BENCHMARK REPORT (Subject s06 - Independent Test)")
    print("=" * 60)
    print(f"MediaPipe Baseline Z-MAE : {base_z_mae:.2f} mm")
    print(f"Proposed Model Refined   : {ref_z_mae:.2f} mm")
    print(f"Z-Depth Error Reduction  : {improvement:.2f}% improvement ⭐")
    print("=" * 60)


if __name__ == "__main__":
    main()
