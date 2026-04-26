from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from dataset import (
    build_full_chord_vocab,
    load_fold_split,
    load_processed_npz,
)
from models import HyperParameters, build_model


@dataclass
class EvalCheckpointConfig:
    root_dir: str
    checkpoint_path: str
    model_type: str
    label_mode: str
    fold_id: int = 0
    split: str = "test"
    embed_size: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.3
    n_steps: int = 128
    stride: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def _first_processed_npz(root_dir: str) -> Path:
    processed_dir = Path(root_dir) / "processed"
    try:
        return next(processed_dir.glob("*.npz"))
    except StopIteration as exc:
        raise FileNotFoundError(f"No processed npz files found in {processed_dir}") from exc


def infer_input_dim(root_dir: str) -> int:
    sample = load_processed_npz(str(_first_processed_npz(root_dir)))
    return int(sample["x"].shape[-1])


def load_eval_model(cfg: EvalCheckpointConfig):
    if cfg.label_mode not in {"full_chord", "structured_full_chord"}:
        raise ValueError(
            "Notebook export/eval expects full chord labels. "
            "Use label_mode='full_chord' or 'structured_full_chord'."
        )

    vocab = build_full_chord_vocab(label_mode=cfg.label_mode)
    hp = HyperParameters(
        n_steps=cfg.n_steps,
        input_embed_size=cfg.embed_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
    )
    device = torch.device(cfg.device)
    model = build_model(
        model_type=cfg.model_type,
        input_dim=infer_input_dim(cfg.root_dir),
        vocab=vocab,
        hyperparameters=hp,
        dropout_rate=cfg.dropout,
    ).to(device)

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {cfg.checkpoint_path}")

    model.load_state_dict(_normalize_state_dict(checkpoint), strict=True)
    model.eval()
    return model, vocab, device


def resolve_song_ids(root_dir: str, fold_id: int, split: str) -> List[str]:
    split = split.lower()
    if split == "all":
        return sorted(p.stem for p in (Path(root_dir) / "processed").glob("*.npz"))

    fold_json = Path(root_dir) / "splits" / f"fold_{fold_id}.json"
    train_ids, val_ids, test_ids = load_fold_split(str(fold_json))
    if split == "train":
        return list(train_ids)
    if split == "val":
        return list(val_ids)
    if split == "test":
        return list(test_ids)
    raise ValueError(f"Unsupported split: {split}")


def make_window_starts(total_frames: int, n_steps: int, stride: int) -> List[int]:
    if total_frames <= n_steps:
        return [0]
    starts = list(range(0, total_frames - n_steps + 1, stride))
    last_start = total_frames - n_steps
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def apply_signal_decay(x: np.ndarray, final_gain: float) -> np.ndarray:
    envelope = np.linspace(1.0, float(final_gain), num=x.shape[0], dtype=np.float32)[:, None]
    return x * envelope


def apply_feature_noise_std(x: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    if noise_std <= 0:
        return x
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=x.shape).astype(np.float32)
    return x + noise


def apply_feature_noise_snr(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    if not np.isfinite(snr_db):
        return x
    signal_power = float(np.mean(np.square(x.astype(np.float64))))
    if signal_power <= 0:
        return x
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=x.shape).astype(np.float32)
    return x + noise


def _batched(iterable: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start:start + batch_size]


@torch.no_grad()
def predict_song_labels(
    model,
    vocab,
    x: np.ndarray,
    device: torch.device,
    n_steps: int,
    stride: int,
    batch_size: int = 32,
) -> List[str]:
    total_frames, feat_dim = x.shape
    starts = make_window_starts(total_frames, n_steps, stride)
    chord_logits_sum = None
    frame_counts = np.zeros(total_frames, dtype=np.float32)

    for batch_starts in _batched(starts, batch_size):
        batch_x = np.zeros((len(batch_starts), n_steps, feat_dim), dtype=np.float32)
        batch_mask = np.zeros((len(batch_starts), n_steps), dtype=np.float32)
        valid_lengths: List[int] = []

        for batch_idx, start in enumerate(batch_starts):
            end = min(start + n_steps, total_frames)
            valid_len = end - start
            batch_x[batch_idx, :valid_len] = x[start:end]
            batch_mask[batch_idx, :valid_len] = 1.0
            valid_lengths.append(valid_len)

        outputs = model(
            x=torch.from_numpy(batch_x).to(device),
            source_mask=torch.from_numpy(batch_mask).to(device),
            target_mask=torch.from_numpy(batch_mask).to(device),
            slope=1.0,
        )
        batch_logits = outputs["chord_logits"].detach().cpu().numpy().astype(np.float32)
        if chord_logits_sum is None:
            chord_logits_sum = np.zeros((total_frames, batch_logits.shape[-1]), dtype=np.float32)

        for batch_idx, start in enumerate(batch_starts):
            valid_len = valid_lengths[batch_idx]
            chord_logits_sum[start:start + valid_len] += batch_logits[batch_idx, :valid_len]
            frame_counts[start:start + valid_len] += 1.0

    if chord_logits_sum is None:
        raise ValueError("No logits were produced during song prediction")

    frame_counts = np.maximum(frame_counts, 1.0)
    chord_logits = chord_logits_sum / frame_counts[:, None]
    pred_ids = chord_logits.argmax(axis=-1)
    return [vocab.decode(int(idx)) for idx in pred_ids]


def labels_to_lab_lines(labels: Sequence[str], frame_rate: float) -> List[str]:
    if not labels:
        return []

    lines: List[str] = []
    start_idx = 0
    current_label = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] != current_label:
            start_t = start_idx / frame_rate
            end_t = idx / frame_rate
            lines.append(f"{start_t:.6f}\t{end_t:.6f}\t{current_label}")
            start_idx = idx
            current_label = labels[idx]

    start_t = start_idx / frame_rate
    end_t = len(labels) / frame_rate
    lines.append(f"{start_t:.6f}\t{end_t:.6f}\t{current_label}")
    return lines


def save_lab(labels: Sequence[str], frame_rate: float, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(labels_to_lab_lines(labels, frame_rate)) + "\n", encoding="utf-8")


def export_predictions(
    cfg: EvalCheckpointConfig,
    output_dir: str,
    corruption_fn: Optional[Callable[[np.ndarray, str], np.ndarray]] = None,
    batch_size: int = 32,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    model, vocab, device = load_eval_model(cfg)
    song_ids = resolve_song_ids(cfg.root_dir, cfg.fold_id, cfg.split)
    if limit is not None:
        song_ids = song_ids[:limit]

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    for song_id in tqdm(song_ids, desc=output_dir_path.name or "predict", leave=False):
        out_path = output_dir_path / f"{song_id}.lab"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        sample = load_processed_npz(str(Path(cfg.root_dir) / "processed" / f"{song_id}.npz"))
        x = sample["x"]
        if corruption_fn is not None:
            x = corruption_fn(x, song_id)
        labels = predict_song_labels(
            model=model,
            vocab=vocab,
            x=x,
            device=device,
            n_steps=cfg.n_steps,
            stride=cfg.stride,
            batch_size=batch_size,
        )
        save_lab(labels, float(sample["frame_rate"]), str(out_path))
        exported += 1

    return {
        "exported": exported,
        "skipped": skipped,
        "total": len(song_ids),
        "output_dir": str(output_dir_path),
        "vocab_labels": list(vocab.idx_to_chord),
    }


def _load_mir_eval():
    import mir_eval
    return mir_eval


def evaluate_mir_eval_metrics(est_dir: str, ref_dir: str) -> Dict[str, float]:
    mir_eval = _load_mir_eval()
    metric_fns = [
        ("root", mir_eval.chord.root),
        ("thirds", mir_eval.chord.thirds),
        ("majmin", mir_eval.chord.majmin),
        ("triads", mir_eval.chord.triads),
        ("sevenths", mir_eval.chord.sevenths),
        ("tetrads", mir_eval.chord.tetrads),
        ("mirex", mir_eval.chord.mirex),
    ]

    per_song = []
    for est_path in tqdm(sorted(Path(est_dir).glob("*.lab")), desc=Path(est_dir).name or "mir_eval", leave=False):
        ref_path = Path(ref_dir) / est_path.name
        if not ref_path.exists():
            continue
        try:
            ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(str(ref_path))
            est_intervals, est_labels = mir_eval.io.load_labeled_intervals(str(est_path))
            est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals,
                est_labels,
                float(ref_intervals.min()),
                float(ref_intervals.max()),
                mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD,
            )
            intervals, merged_ref, merged_est = mir_eval.util.merge_labeled_intervals(
                ref_intervals,
                ref_labels,
                est_intervals,
                est_labels,
            )
            durations = mir_eval.util.intervals_to_durations(intervals)
        except Exception:
            continue

        scores = {}
        for name, fn in metric_fns:
            comparisons = fn(merged_ref, merged_est)
            scores[name] = float(mir_eval.chord.weighted_accuracy(comparisons, durations))
        per_song.append(scores)

    if not per_song:
        return {name: float("nan") for name, _ in metric_fns}

    return {
        name: float(np.mean([song[name] for song in per_song]))
        for name, _ in metric_fns
    }


def build_mir_eval_vocab_keys(vocab_labels: Sequence[str]):
    mir_eval = _load_mir_eval()

    def chord_key(label: str):
        if label is None or label == "" or label == "X":
            return None
        if label == "N":
            return ("N",)
        try:
            root, intervals, bass = mir_eval.chord.encode(label, reduce_extended_chords=False)
        except mir_eval.chord.InvalidChordException:
            return None
        if root < 0:
            return ("N",)
        return (int(root), tuple(int(x) for x in intervals), int(bass))

    vocab_keys = set()
    for label in vocab_labels:
        key = chord_key(label)
        if key is not None:
            vocab_keys.add(key)
    return chord_key, vocab_keys


def rasterize_labels(intervals, labels, n_frames: int, fps: float) -> np.ndarray:
    out = np.full(n_frames, "N", dtype=object)
    for (start, end), label in zip(intervals, labels):
        frame_start = max(0, int(round(float(start) * fps)))
        frame_end = min(n_frames, int(round(float(end) * fps)))
        if frame_end > frame_start:
            out[frame_start:frame_end] = label
    return out


def evaluate_large_vocabulary(
    est_dir: str,
    ref_dir: str,
    vocab_labels: Sequence[str],
    fps: float = 22050.0 / 512.0,
) -> Dict[str, float]:
    mir_eval = _load_mir_eval()
    chord_key, vocab_keys = build_mir_eval_vocab_keys(vocab_labels)

    total = 0
    correct = 0
    cls_correct = {key: 0 for key in vocab_keys}
    cls_total = {key: 0 for key in vocab_keys}

    for est_path in tqdm(sorted(Path(est_dir).glob("*.lab")), desc=Path(est_dir).name or "large_vocab", leave=False):
        ref_path = Path(ref_dir) / est_path.name
        if not ref_path.exists():
            continue
        try:
            ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(str(ref_path))
            est_intervals, est_labels = mir_eval.io.load_labeled_intervals(str(est_path))
        except Exception:
            continue

        end_time = min(float(ref_intervals[-1, 1]), float(est_intervals[-1, 1]))
        if end_time <= 0:
            continue

        n_frames = int(end_time * fps)
        ref_frames = rasterize_labels(ref_intervals, ref_labels, n_frames, fps)
        est_frames = rasterize_labels(est_intervals, est_labels, n_frames, fps)

        key_cache: Dict[str, object] = {}

        def get_key(label: str):
            if label not in key_cache:
                key_cache[label] = chord_key(label)
            return key_cache[label]

        for ref_label, est_label in zip(ref_frames, est_frames):
            ref_key = get_key(ref_label)
            if ref_key is None or ref_key not in vocab_keys:
                continue
            est_key = get_key(est_label)
            total += 1
            cls_total[ref_key] += 1
            if est_key == ref_key:
                correct += 1
                cls_correct[ref_key] += 1

    seen = [key for key in vocab_keys if cls_total[key] > 0]
    frame_acc = (correct / total) if total > 0 else float("nan")
    class_acc = (
        float(np.mean([cls_correct[key] / cls_total[key] for key in seen]))
        if seen else float("nan")
    )
    return {
        "frame_acc": frame_acc,
        "class_acc": class_acc,
        "frames": int(total),
        "classes_seen": int(len(seen)),
        "vocab_key_count": int(len(vocab_keys)),
    }
