from typing import Optional, Dict
import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from dataset import *
from model import *

def masked_cross_entropy(
    logits: torch.Tensor,   # [B, T, C]
    targets: torch.Tensor,  # [B, T]
    mask: torch.Tensor,     # [B, T]
) -> torch.Tensor:
    """
    Cross-entropy only on valid (non-padded) positions.
    """
    B, T, C = logits.shape

    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    mask = mask.reshape(B * T).float()

    per_token_loss = F.cross_entropy(logits, targets, reduction="none")  # [B*T]
    masked_loss = per_token_loss * mask

    loss = masked_loss.sum() / (mask.sum() + 1e-8)
    return loss


def masked_bce_with_logits(
    logits: torch.Tensor,   # [B, T]
    targets: torch.Tensor,  # [B, T], float or int {0,1}
    mask: torch.Tensor,     # [B, T]
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCE for boundary prediction, only on valid positions.
    """
    targets = targets.float()
    mask = mask.float()

    per_pos_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )  # [B, T]

    masked_loss = per_pos_loss * mask
    loss = masked_loss.sum() / (mask.sum() + 1e-8)
    return loss


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    chord_targets: torch.Tensor,          # [B, T]
    chord_change_targets: torch.Tensor,   # [B, T]
    mask: torch.Tensor,                   # [B, T]
    chord_loss_weight: float = 1.0,
    change_loss_weight: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    chord_loss = masked_cross_entropy(
        logits=outputs["chord_logits"],
        targets=chord_targets,
        mask=mask,
    )

    change_loss = masked_bce_with_logits(
        logits=outputs["chord_change_logits"],
        targets=chord_change_targets,
        mask=mask,
        pos_weight=pos_weight,
    )

    total_loss = chord_loss_weight * chord_loss + change_loss_weight * change_loss

    return {
        "loss": total_loss,
        "chord_loss": chord_loss,
        "change_loss": change_loss,
    }

# =========================================================
# Training / evaluation steps
# =========================================================

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    slope: float = 1.0,
    chord_loss_weight: float = 1.0,
    change_loss_weight: float = 1.0,
    grad_clip: Optional[float] = None,
    pos_weight: Optional[torch.Tensor] = None,
):
    """
    batch should contain:
      x                     [B, T, F]   input features (e.g. CQT)
      chord_targets         [B, T]      class ids
      chord_change_targets  [B, T]      0/1 boundary labels
      mask                  [B, T]      1 valid, 0 padding
    """
    model.train()
    optimizer.zero_grad()

    x = batch["x"].to(device)
    chord_targets = batch["chord_targets"].to(device)
    chord_change_targets = batch["chord_change_targets"].to(device)
    mask = batch["mask"].to(device)

    outputs = model(
        x=x,
        source_mask=mask,
        target_mask=mask,
        slope=slope,
    )

    losses = compute_losses(
        outputs=outputs,
        chord_targets=chord_targets,
        chord_change_targets=chord_change_targets,
        mask=mask,
        chord_loss_weight=chord_loss_weight,
        change_loss_weight=change_loss_weight,
        pos_weight=pos_weight.to(device) if pos_weight is not None else None,
    )

    losses["loss"].backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    with torch.no_grad():
        pred_chords = outputs["chord_logits"].argmax(dim=-1)  # [B, T]
        correct = ((pred_chords == chord_targets) & mask.bool()).float().sum()
        total = mask.float().sum()
        chord_acc = correct / (total + 1e-8)

        pred_change = (torch.sigmoid(outputs["chord_change_logits"]) >= 0.5).long()
        change_correct = ((pred_change == chord_change_targets.long()) & mask.bool()).float().sum()
        change_acc = change_correct / (total + 1e-8)

    return {
        "loss": losses["loss"].item(),
        "chord_loss": losses["chord_loss"].item(),
        "change_loss": losses["change_loss"].item(),
        "chord_acc": chord_acc.item(),
        "change_acc": change_acc.item(),
    }


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    slope: float = 1.0,
    chord_loss_weight: float = 1.0,
    change_loss_weight: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
):
    model.eval()

    x = batch["x"].to(device)
    chord_targets = batch["chord_targets"].to(device)
    chord_change_targets = batch["chord_change_targets"].to(device)
    mask = batch["mask"].to(device)

    outputs = model(
        x=x,
        source_mask=mask,
        target_mask=mask,
        slope=slope,
    )

    losses = compute_losses(
        outputs=outputs,
        chord_targets=chord_targets,
        chord_change_targets=chord_change_targets,
        mask=mask,
        chord_loss_weight=chord_loss_weight,
        change_loss_weight=change_loss_weight,
        pos_weight=pos_weight.to(device) if pos_weight is not None else None,
    )

    pred_chords = outputs["chord_logits"].argmax(dim=-1)
    correct = ((pred_chords == chord_targets) & mask.bool()).float().sum()
    total = mask.float().sum()
    chord_acc = correct / (total + 1e-8)

    pred_change = (torch.sigmoid(outputs["chord_change_logits"]) >= 0.5).long()
    change_correct = ((pred_change == chord_change_targets.long()) & mask.bool()).float().sum()
    change_acc = change_correct / (total + 1e-8)

    return {
        "loss": losses["loss"].item(),
        "chord_loss": losses["chord_loss"].item(),
        "change_loss": losses["change_loss"].item(),
        "chord_acc": chord_acc.item(),
        "change_acc": change_acc.item(),
    }


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    slope=1.0,
    chord_loss_weight=1.0,
    change_loss_weight=0.5,
    grad_clip=None,
):
    metrics_all = []
    total_items = len(train_loader.dataset) if hasattr(train_loader, "dataset") else None

    progress = tqdm(
        total=total_items,
        desc="train",
        unit="sample",
        leave=False,
    )

    try:
        for batch in train_loader:
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                slope=slope,
                chord_loss_weight=chord_loss_weight,
                change_loss_weight=change_loss_weight,
                grad_clip=grad_clip,
            )
            metrics_all.append(metrics)

            batch_size = batch["x"].shape[0]
            progress.update(batch_size)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                chord_acc=f"{metrics['chord_acc']:.4f}",
                change_acc=f"{metrics['change_acc']:.4f}",
            )
    finally:
        progress.close()

    return {
        "loss": sum(m["loss"] for m in metrics_all) / len(metrics_all),
        "chord_loss": sum(m["chord_loss"] for m in metrics_all) / len(metrics_all),
        "change_loss": sum(m["change_loss"] for m in metrics_all) / len(metrics_all),
        "chord_acc": sum(m["chord_acc"] for m in metrics_all) / len(metrics_all),
        "change_acc": sum(m["change_acc"] for m in metrics_all) / len(metrics_all),
    }


def eval_one_epoch(
    model,
    data_loader,
    device,
    slope=1.0,
    chord_loss_weight=1.0,
    change_loss_weight=0.5,
):
    metrics_all = []
    total_items = len(data_loader.dataset) if hasattr(data_loader, "dataset") else None

    progress = tqdm(
        total=total_items,
        desc="eval",
        unit="sample",
        leave=False,
    )

    try:
        for batch in data_loader:
            metrics = eval_step(
                model=model,
                batch=batch,
                device=device,
                slope=slope,
                chord_loss_weight=chord_loss_weight,
                change_loss_weight=change_loss_weight,
            )
            metrics_all.append(metrics)

            batch_size = batch["x"].shape[0]
            progress.update(batch_size)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                chord_acc=f"{metrics['chord_acc']:.4f}",
                change_acc=f"{metrics['change_acc']:.4f}",
            )
    finally:
        progress.close()

    return {
        "loss": sum(m["loss"] for m in metrics_all) / len(metrics_all),
        "chord_loss": sum(m["chord_loss"] for m in metrics_all) / len(metrics_all),
        "change_loss": sum(m["change_loss"] for m in metrics_all) / len(metrics_all),
        "chord_acc": sum(m["chord_acc"] for m in metrics_all) / len(metrics_all),
        "change_acc": sum(m["change_acc"] for m in metrics_all) / len(metrics_all),
    }
    
    
def run_one_fold(
    fold_json_path,
    root_dir,
    device,
    args,
):
    cfg = ProcessedChordConfig(
        root_dir=root_dir,
        n_steps=128,
        stride=64,
        batch_size=8,
        num_workers=0,
    )

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, vocab = \
        build_processed_loaders(cfg, fold_json_path)

    if len(train_dataset) == 0:
        raise ValueError(f"No training windows found for {fold_json_path}")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation windows found for {fold_json_path}")
    if len(test_dataset) == 0:
        raise ValueError(f"No test windows found for {fold_json_path}")
    
    # infer input_dim from one batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["x"].shape[-1]
    n_chords = vocab.size
    
    print(f"Vocab size {n_chords}")

    hp = HyperParameters(
        n_steps=cfg.n_steps,
        input_embed_size=256,
        n_layers=4,
        n_heads=8,
    )

    model = HTv2ChordModel(
        input_dim=input_dim,
        n_chords=n_chords,
        hyperparameters=hp,
        dropout_rate=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    epoch_progress = tqdm(
        range(args.max_n_epochs),
        desc=os.path.basename(fold_json_path),
        unit="epoch",
    )
    
    ckpt_dir = os.path.join(root_dir, "checkpoints", args.experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    fold_name = os.path.splitext(os.path.basename(fold_json_path))[0]
    best_ckpt_path = os.path.join(ckpt_dir, f"{fold_name}_best.pt")

    for epoch in epoch_progress:
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            slope=1.0,
            chord_loss_weight=1.0,
            change_loss_weight=0,
            grad_clip=args.grad_clip,
        )

        val_metrics = eval_one_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            slope=1.0,
            chord_loss_weight=1.0,
            change_loss_weight=0,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_progress.set_postfix(
            lr=f"{current_lr:.2e}",
            train_loss=f"{train_metrics['loss']:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            val_chord_acc=f"{val_metrics['chord_acc']:.4f}",
        )

        tqdm.write(
            f"{os.path.basename(fold_json_path)} | "
            f"Epoch {epoch+1:02d} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_chord_acc={val_metrics['chord_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, best_ckpt_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            tqdm.write(f"Early stopping on {os.path.basename(fold_json_path)}")
            break

    # load best model for this fold
    model.load_state_dict(best_state_dict)

    test_metrics = eval_one_epoch(
        model=model,
        data_loader=test_loader,
        device=device,
        slope=1.0,
        chord_loss_weight=1.0,
        change_loss_weight=0.5,
    )

    return {
        "fold_file": os.path.basename(fold_json_path),
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_chord_loss": test_metrics["chord_loss"],
        "test_change_loss": test_metrics["change_loss"],
        "test_chord_acc": test_metrics["chord_acc"],
        "test_change_acc": test_metrics["change_acc"],
        "vocab_size": vocab.size,
        "num_train_windows": len(train_dataset),
        "num_val_windows": len(val_dataset),
        "num_test_windows": len(test_dataset),
    }
    
def run_cross_validation(
    root_dir,
    device,
    args
):
    splits_dir = os.path.join(root_dir, "splits")

    fold_files = [
        os.path.join(splits_dir, f"fold_{i}.json")
        for i in range(5)
    ]

    all_results = []

    for fold_json_path in tqdm(fold_files, desc="folds", unit="fold"):
        tqdm.write("=" * 80)
        tqdm.write(f"Running {os.path.basename(fold_json_path)}")
        tqdm.write("=" * 80)

        fold_result = run_one_fold(
            fold_json_path=fold_json_path,
            root_dir=root_dir,
            device=device,
            args=args
        )
        all_results.append(fold_result)

        tqdm.write(
            f"Finished {fold_result['fold_file']} | "
            f"test_loss={fold_result['test_loss']:.4f} | "
            f"test_chord_acc={fold_result['test_chord_acc']:.4f} | "
            f"test_change_acc={fold_result['test_change_acc']:.4f}"
        )

    mean_test_loss = np.mean([r["test_loss"] for r in all_results])
    std_test_loss = np.std([r["test_loss"] for r in all_results])

    mean_test_chord_acc = np.mean([r["test_chord_acc"] for r in all_results])
    std_test_chord_acc = np.std([r["test_chord_acc"] for r in all_results])

    mean_test_change_acc = np.mean([r["test_change_acc"] for r in all_results])
    std_test_change_acc = np.std([r["test_change_acc"] for r in all_results])

    print("\n" + "=" * 80)
    print("Cross-validation summary")
    print("=" * 80)
    for r in all_results:
        print(
            f"{r['fold_file']}: "
            f"test_loss={r['test_loss']:.4f}, "
            f"test_chord_acc={r['test_chord_acc']:.4f}, "
            f"test_change_acc={r['test_change_acc']:.4f}"
        )

    print("-" * 80)
    print(f"mean_test_loss      = {mean_test_loss:.4f} ± {std_test_loss:.4f}")
    print(f"mean_test_chord_acc = {mean_test_chord_acc:.4f} ± {std_test_chord_acc:.4f}")
    print(f"mean_test_change_acc= {mean_test_change_acc:.4f} ± {std_test_change_acc:.4f}")

    return all_results
