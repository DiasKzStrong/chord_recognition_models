from typing import Optional, Dict, List
import copy
import os

import numpy as np
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
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    B, T, C = logits.shape

    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    mask = mask.reshape(B * T).float()

    per_token_loss = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    masked_loss = per_token_loss * mask
    return masked_loss.sum() / (mask.sum() + 1e-8)


def masked_bce_with_logits(
    logits: torch.Tensor,   # [B, T]
    targets: torch.Tensor,  # [B, T], float or int {0,1}
    mask: torch.Tensor,     # [B, T]
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    targets = targets.float()
    mask = mask.float()

    per_pos_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    masked_loss = per_pos_loss * mask
    return masked_loss.sum() / (mask.sum() + 1e-8)


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    chord_targets: torch.Tensor,
    chord_change_targets: torch.Tensor,
    mask: torch.Tensor,
    chord_loss_weight: float = 1.0,
    change_loss_weight: float = 1.0,
    change_pos_weight: Optional[torch.Tensor] = None,
    chord_class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> Dict[str, torch.Tensor]:
    chord_loss = masked_cross_entropy(
        logits=outputs["chord_logits"],
        targets=chord_targets,
        mask=mask,
        class_weights=chord_class_weights,
        label_smoothing=label_smoothing,
    )

    change_loss = masked_bce_with_logits(
        logits=outputs["chord_change_logits"],
        targets=chord_change_targets,
        mask=mask,
        pos_weight=change_pos_weight,
    )

    total_loss = chord_loss_weight * chord_loss + change_loss_weight * change_loss

    return {
        "loss": total_loss,
        "chord_loss": chord_loss,
        "change_loss": change_loss,
    }


def _metric_counts(
    outputs: Dict[str, torch.Tensor],
    chord_targets: torch.Tensor,
    chord_change_targets: torch.Tensor,
    mask: torch.Tensor,
    n_chords: int,
) -> Dict[str, object]:
    valid = mask.bool()
    total = int(valid.sum().item())

    pred_chords = outputs["chord_logits"].argmax(dim=-1)
    chord_correct_mask = (pred_chords == chord_targets) & valid
    chord_correct = int(chord_correct_mask.sum().item())

    valid_targets = chord_targets[valid]
    per_class_total = torch.bincount(valid_targets, minlength=n_chords).detach().cpu().numpy()
    per_class_correct = torch.bincount(
        chord_targets[chord_correct_mask],
        minlength=n_chords,
    ).detach().cpu().numpy()

    pred_change = (torch.sigmoid(outputs["chord_change_logits"]) >= 0.5)
    target_change = chord_change_targets.bool()
    change_correct = int(((pred_change == target_change) & valid).sum().item())
    change_tp = int((pred_change & target_change & valid).sum().item())
    change_fp = int((pred_change & ~target_change & valid).sum().item())
    change_fn = int((~pred_change & target_change & valid).sum().item())

    return {
        "total": total,
        "chord_correct": chord_correct,
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "change_correct": change_correct,
        "change_tp": change_tp,
        "change_fp": change_fp,
        "change_fn": change_fn,
    }


def _empty_epoch_state(n_chords: int) -> Dict[str, object]:
    return {
        "loss_sum": 0.0,
        "chord_loss_sum": 0.0,
        "change_loss_sum": 0.0,
        "total": 0,
        "chord_correct": 0,
        "per_class_total": np.zeros(n_chords, dtype=np.float64),
        "per_class_correct": np.zeros(n_chords, dtype=np.float64),
        "change_correct": 0,
        "change_tp": 0,
        "change_fp": 0,
        "change_fn": 0,
    }


def _update_epoch_state(state: Dict[str, object], metrics: Dict[str, object]) -> None:
    total = int(metrics["total"])
    state["loss_sum"] += float(metrics["loss"]) * total
    state["chord_loss_sum"] += float(metrics["chord_loss"]) * total
    state["change_loss_sum"] += float(metrics["change_loss"]) * total
    state["total"] += total
    state["chord_correct"] += int(metrics["chord_correct"])
    state["per_class_total"] += metrics["per_class_total"]
    state["per_class_correct"] += metrics["per_class_correct"]
    state["change_correct"] += int(metrics["change_correct"])
    state["change_tp"] += int(metrics["change_tp"])
    state["change_fp"] += int(metrics["change_fp"])
    state["change_fn"] += int(metrics["change_fn"])


def _summarize_epoch_state(state: Dict[str, object], vocab: FixedChordVocab) -> Dict[str, object]:
    total = max(int(state["total"]), 1)
    per_class_total = state["per_class_total"]
    per_class_correct = state["per_class_correct"]
    seen = per_class_total > 0
    per_class_acc = np.zeros_like(per_class_total, dtype=np.float64)
    per_class_acc[seen] = per_class_correct[seen] / per_class_total[seen]

    tp = int(state["change_tp"])
    fp = int(state["change_fp"])
    fn = int(state["change_fn"])
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "loss": state["loss_sum"] / total,
        "chord_loss": state["chord_loss_sum"] / total,
        "change_loss": state["change_loss_sum"] / total,
        "chord_acc": int(state["chord_correct"]) / total,
        "macro_chord_acc": float(per_class_acc[seen].mean()) if seen.any() else 0.0,
        "per_class_acc": {
            vocab.decode(i): float(per_class_acc[i])
            for i in range(vocab.size)
            if per_class_total[i] > 0
        },
        "change_acc": int(state["change_correct"]) / total,
        "change_precision": precision,
        "change_recall": recall,
        "change_f1": f1,
    }


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_chords: int,
    slope: float = 1.0,
    chord_loss_weight: float = 1.0,
    change_loss_weight: float = 1.0,
    grad_clip: Optional[float] = None,
    change_pos_weight: Optional[torch.Tensor] = None,
    chord_class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    boundary_teacher_forcing_prob: float = 0.0,
):
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
        chord_change_targets=chord_change_targets,
        boundary_teacher_forcing_prob=boundary_teacher_forcing_prob,
    )

    losses = compute_losses(
        outputs=outputs,
        chord_targets=chord_targets,
        chord_change_targets=chord_change_targets,
        mask=mask,
        chord_loss_weight=chord_loss_weight,
        change_loss_weight=change_loss_weight,
        change_pos_weight=change_pos_weight.to(device) if change_pos_weight is not None else None,
        chord_class_weights=chord_class_weights.to(device) if chord_class_weights is not None else None,
        label_smoothing=label_smoothing,
    )

    losses["loss"].backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    with torch.no_grad():
        counts = _metric_counts(outputs, chord_targets, chord_change_targets, mask, n_chords)

    counts.update({
        "loss": losses["loss"].item(),
        "chord_loss": losses["chord_loss"].item(),
        "change_loss": losses["change_loss"].item(),
    })
    return counts


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    n_chords: int,
    slope: float = 1.0,
    chord_loss_weight: float = 1.0,
    change_loss_weight: float = 1.0,
    change_pos_weight: Optional[torch.Tensor] = None,
    chord_class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
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
        change_pos_weight=change_pos_weight.to(device) if change_pos_weight is not None else None,
        chord_class_weights=chord_class_weights.to(device) if chord_class_weights is not None else None,
        label_smoothing=label_smoothing,
    )

    counts = _metric_counts(outputs, chord_targets, chord_change_targets, mask, n_chords)
    counts.update({
        "loss": losses["loss"].item(),
        "chord_loss": losses["chord_loss"].item(),
        "change_loss": losses["change_loss"].item(),
    })
    return counts


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    vocab,
    slope=1.0,
    chord_loss_weight=1.0,
    change_loss_weight=0.1,
    grad_clip=None,
    change_pos_weight=None,
    chord_class_weights=None,
    label_smoothing=0.0,
    boundary_teacher_forcing_prob=0.0,
):
    state = _empty_epoch_state(vocab.size)
    total_items = len(train_loader.dataset) if hasattr(train_loader, "dataset") else None

    progress = tqdm(total=total_items, desc="train", unit="sample", leave=False)

    try:
        for batch in train_loader:
            metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                n_chords=vocab.size,
                slope=slope,
                chord_loss_weight=chord_loss_weight,
                change_loss_weight=change_loss_weight,
                grad_clip=grad_clip,
                change_pos_weight=change_pos_weight,
                chord_class_weights=chord_class_weights,
                label_smoothing=label_smoothing,
                boundary_teacher_forcing_prob=boundary_teacher_forcing_prob,
            )
            _update_epoch_state(state, metrics)

            batch_size = batch["x"].shape[0]
            batch_total = max(int(metrics["total"]), 1)
            progress.update(batch_size)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                chord_acc=f"{metrics['chord_correct'] / batch_total:.4f}",
                change_f1=_format_change_f1(metrics),
            )
    finally:
        progress.close()

    return _summarize_epoch_state(state, vocab)


def eval_one_epoch(
    model,
    data_loader,
    device,
    vocab,
    slope=1.0,
    chord_loss_weight=1.0,
    change_loss_weight=0.1,
    change_pos_weight=None,
    chord_class_weights=None,
    label_smoothing=0.0,
):
    state = _empty_epoch_state(vocab.size)
    total_items = len(data_loader.dataset) if hasattr(data_loader, "dataset") else None

    progress = tqdm(total=total_items, desc="eval", unit="sample", leave=False)

    try:
        for batch in data_loader:
            metrics = eval_step(
                model=model,
                batch=batch,
                device=device,
                n_chords=vocab.size,
                slope=slope,
                chord_loss_weight=chord_loss_weight,
                change_loss_weight=change_loss_weight,
                change_pos_weight=change_pos_weight,
                chord_class_weights=chord_class_weights,
                label_smoothing=label_smoothing,
            )
            _update_epoch_state(state, metrics)

            batch_size = batch["x"].shape[0]
            batch_total = max(int(metrics["total"]), 1)
            progress.update(batch_size)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                chord_acc=f"{metrics['chord_correct'] / batch_total:.4f}",
                change_f1=_format_change_f1(metrics),
            )
    finally:
        progress.close()

    return _summarize_epoch_state(state, vocab)


def _format_change_f1(metrics: Dict[str, object]) -> str:
    tp = int(metrics["change_tp"])
    fp = int(metrics["change_fp"])
    fn = int(metrics["change_fn"])
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f"{f1:.4f}"


def compute_train_statistics(dataset, vocab):
    class_counts = np.zeros(vocab.size, dtype=np.float64)
    change_pos = 0.0
    valid_total = 0.0

    for item in dataset.items:
        mask = item["mask"].astype(bool)
        targets = item["chord_targets"][mask]
        changes = item["chord_change_targets"][mask]
        class_counts += np.bincount(targets, minlength=vocab.size)
        change_pos += float(changes.sum())
        valid_total += float(mask.sum())

    majority_idx = int(class_counts.argmax())
    majority_acc = float(class_counts[majority_idx] / max(class_counts.sum(), 1.0))
    change_rate = float(change_pos / max(valid_total, 1.0))

    return {
        "class_counts": class_counts,
        "majority_class": vocab.decode(majority_idx),
        "majority_acc": majority_acc,
        "change_pos": change_pos,
        "change_neg": valid_total - change_pos,
        "change_rate": change_rate,
    }


def make_class_weights(class_counts: np.ndarray, mode: str) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    if mode != "inverse_sqrt":
        raise ValueError(f"Unsupported class weighting mode: {mode}")

    counts = np.maximum(class_counts.astype(np.float64), 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.3, 3.0)
    return torch.tensor(weights, dtype=torch.float32)


def make_change_pos_weight(change_pos: float, change_neg: float, max_weight: float) -> torch.Tensor:
    if change_pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    weight = change_neg / change_pos
    if max_weight > 0:
        weight = min(weight, max_weight)
    return torch.tensor(float(weight), dtype=torch.float32)


def teacher_forcing_prob_for_epoch(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 0.0
    if epoch >= warmup_epochs:
        return 0.0
    return 1.0 - (epoch / warmup_epochs)


def print_fold_data_summary(fold_name: str, train_dataset, val_dataset, test_dataset, stats, vocab) -> None:
    class_counts = stats["class_counts"]
    top_ids = np.argsort(class_counts)[::-1][:8]
    top_classes = ", ".join(
        f"{vocab.decode(int(i))}:{class_counts[int(i)] / max(class_counts.sum(), 1.0):.3f}"
        for i in top_ids
    )
    print(
        f"{fold_name} data | "
        f"train_windows={len(train_dataset)} | val_windows={len(val_dataset)} | test_windows={len(test_dataset)} | "
        f"majority={stats['majority_class']}:{stats['majority_acc']:.4f} | "
        f"change_rate={stats['change_rate']:.4f} | top={top_classes}"
    )


def format_worst_classes(per_class_acc: Dict[str, float], limit: int = 6) -> str:
    if not per_class_acc:
        return ""
    items = sorted(per_class_acc.items(), key=lambda kv: kv[1])[:limit]
    return ", ".join(f"{name}:{acc:.3f}" for name, acc in items)


def run_one_fold(
    fold_json_path,
    root_dir,
    device,
    args,
):
    cfg = ProcessedChordConfig(
        root_dir=root_dir,
        n_steps=args.n_steps,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=args.augment,
        noise_std=args.noise_std,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        time_mask_width=args.time_mask_width,
        freq_mask_width=args.freq_mask_width,
        pitch_shift_bins=args.pitch_shift_bins,
        use_signal_decay=args.use_signal_decay,
        signal_decay_min=args.signal_decay_min,
        signal_decay_max=args.signal_decay_max,
    )

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, vocab = \
        build_processed_loaders(cfg, fold_json_path)

    if len(train_dataset) == 0:
        raise ValueError(f"No training windows found for {fold_json_path}")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation windows found for {fold_json_path}")
    if len(test_dataset) == 0:
        raise ValueError(f"No test windows found for {fold_json_path}")

    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["x"].shape[-1]
    n_chords = vocab.size

    fold_name = os.path.splitext(os.path.basename(fold_json_path))[0]
    print(f"Vocab size {n_chords}")

    stats = compute_train_statistics(train_dataset, vocab)
    print_fold_data_summary(fold_name, train_dataset, val_dataset, test_dataset, stats, vocab)

    chord_class_weights = make_class_weights(stats["class_counts"], args.class_weighting)
    change_pos_weight = make_change_pos_weight(
        stats["change_pos"],
        stats["change_neg"],
        args.max_change_pos_weight,
    )
    if chord_class_weights is not None:
        print(f"{fold_name} class weights enabled: mode={args.class_weighting}")
    print(f"{fold_name} boundary pos_weight={change_pos_weight.item():.2f}")

    hp = HyperParameters(
        n_steps=cfg.n_steps,
        input_embed_size=args.embed_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )

    model = HTv2ChordModel(
        input_dim=input_dim,
        n_chords=n_chords,
        hyperparameters=hp,
        dropout_rate=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val_score = -float("inf")
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    epoch_progress = tqdm(range(args.max_n_epochs), desc=os.path.basename(fold_json_path), unit="epoch")

    ckpt_dir = os.path.join(root_dir, "checkpoints", args.experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"{fold_name}_best.pt")

    for epoch in epoch_progress:
        boundary_tf_prob = teacher_forcing_prob_for_epoch(epoch, args.boundary_teacher_forcing_epochs)

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            vocab=vocab,
            slope=1.0,
            chord_loss_weight=1.0,
            change_loss_weight=args.change_loss_weight,
            grad_clip=args.grad_clip,
            change_pos_weight=change_pos_weight,
            chord_class_weights=chord_class_weights,
            label_smoothing=args.label_smoothing,
            boundary_teacher_forcing_prob=boundary_tf_prob,
        )

        val_metrics = eval_one_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            vocab=vocab,
            slope=1.0,
            chord_loss_weight=1.0,
            change_loss_weight=args.change_loss_weight,
            change_pos_weight=change_pos_weight,
            chord_class_weights=chord_class_weights,
            label_smoothing=args.label_smoothing,
        )

        val_score = float(val_metrics[args.checkpoint_metric])
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_progress.set_postfix(
            lr=f"{current_lr:.2e}",
            train_loss=f"{train_metrics['loss']:.4f}",
            val_acc=f"{val_metrics['chord_acc']:.4f}",
            val_macro=f"{val_metrics['macro_chord_acc']:.4f}",
        )

        tqdm.write(
            f"{os.path.basename(fold_json_path)} | "
            f"Epoch {epoch+1:02d} | "
            f"lr={current_lr:.2e} | "
            f"tf={boundary_tf_prob:.2f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_chord_acc={train_metrics['chord_acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_chord_acc={val_metrics['chord_acc']:.4f} | "
            f"val_macro_chord_acc={val_metrics['macro_chord_acc']:.4f} | "
            f"val_change_f1={val_metrics['change_f1']:.4f}"
        )

        improved = (val_score > best_val_score + args.min_delta)
        if improved or (val_score == best_val_score and val_metrics["loss"] < best_val_loss):
            best_val_score = val_score
            best_val_loss = val_metrics["loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, best_ckpt_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            tqdm.write(f"Early stopping on {os.path.basename(fold_json_path)}")
            break

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state_dict)

    test_metrics = eval_one_epoch(
        model=model,
        data_loader=test_loader,
        device=device,
        vocab=vocab,
        slope=1.0,
        chord_loss_weight=1.0,
        change_loss_weight=args.change_loss_weight,
        change_pos_weight=change_pos_weight,
        chord_class_weights=chord_class_weights,
        label_smoothing=args.label_smoothing,
    )
    print(f"{fold_name} worst test classes: {format_worst_classes(test_metrics['per_class_acc'])}")

    return {
        "fold_file": os.path.basename(fold_json_path),
        "best_val_score": best_val_score,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_chord_loss": test_metrics["chord_loss"],
        "test_change_loss": test_metrics["change_loss"],
        "test_chord_acc": test_metrics["chord_acc"],
        "test_macro_chord_acc": test_metrics["macro_chord_acc"],
        "test_change_acc": test_metrics["change_acc"],
        "test_change_precision": test_metrics["change_precision"],
        "test_change_recall": test_metrics["change_recall"],
        "test_change_f1": test_metrics["change_f1"],
        "test_per_class_acc": test_metrics["per_class_acc"],
        "vocab_size": vocab.size,
        "num_train_windows": len(train_dataset),
        "num_val_windows": len(val_dataset),
        "num_test_windows": len(test_dataset),
    }


def run_cross_validation(root_dir, device, args):
    splits_dir = os.path.join(root_dir, "splits")

    if args.fold_ids:
        fold_indices = [int(x.strip()) for x in args.fold_ids.split(",") if x.strip()]
    else:
        fold_indices = list(range(args.num_folds))

    fold_files = [os.path.join(splits_dir, f"fold_{i}.json") for i in fold_indices]
    all_results = []

    for fold_json_path in tqdm(fold_files, desc="folds", unit="fold"):
        tqdm.write("=" * 80)
        tqdm.write(f"Running {os.path.basename(fold_json_path)}")
        tqdm.write("=" * 80)

        fold_result = run_one_fold(
            fold_json_path=fold_json_path,
            root_dir=root_dir,
            device=device,
            args=args,
        )
        all_results.append(fold_result)

        tqdm.write(
            f"Finished {fold_result['fold_file']} | "
            f"test_loss={fold_result['test_loss']:.4f} | "
            f"test_chord_acc={fold_result['test_chord_acc']:.4f} | "
            f"test_macro_chord_acc={fold_result['test_macro_chord_acc']:.4f} | "
            f"test_change_f1={fold_result['test_change_f1']:.4f}"
        )

    mean_test_loss = np.mean([r["test_loss"] for r in all_results])
    std_test_loss = np.std([r["test_loss"] for r in all_results])
    mean_test_chord_acc = np.mean([r["test_chord_acc"] for r in all_results])
    std_test_chord_acc = np.std([r["test_chord_acc"] for r in all_results])
    mean_test_macro_chord_acc = np.mean([r["test_macro_chord_acc"] for r in all_results])
    std_test_macro_chord_acc = np.std([r["test_macro_chord_acc"] for r in all_results])
    mean_test_change_f1 = np.mean([r["test_change_f1"] for r in all_results])
    std_test_change_f1 = np.std([r["test_change_f1"] for r in all_results])

    print("\n" + "=" * 80)
    print("Cross-validation summary")
    print("=" * 80)
    for r in all_results:
        print(
            f"{r['fold_file']}: "
            f"test_loss={r['test_loss']:.4f}, "
            f"test_chord_acc={r['test_chord_acc']:.4f}, "
            f"test_macro_chord_acc={r['test_macro_chord_acc']:.4f}, "
            f"test_change_f1={r['test_change_f1']:.4f}"
        )

    print("-" * 80)
    print(f"mean_test_loss            = {mean_test_loss:.4f} ± {std_test_loss:.4f}")
    print(f"mean_test_chord_acc       = {mean_test_chord_acc:.4f} ± {std_test_chord_acc:.4f}")
    print(f"mean_test_macro_chord_acc = {mean_test_macro_chord_acc:.4f} ± {std_test_macro_chord_acc:.4f}")
    print(f"mean_test_change_f1       = {mean_test_change_f1:.4f} ± {std_test_change_f1:.4f}")

    return all_results
