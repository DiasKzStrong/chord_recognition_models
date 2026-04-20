import os
import json
from dataclasses import dataclass
from typing import Dict, List
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


TARGET_CLASSES = [
    "Others",
    "min/b7",
    "min/2",
    "maj/b7",
    "maj/2",
    "sus4(b7)",
    "sus2",
    "sus4",
    "13",
    "11",
    "min9",
    "9",
    "maj9",
    "dim7",
    "hdim7",
    "min7",
    "7",
    "maj7",
    "min/5",
    "min/b3",
    "maj/5",
    "maj/3",
    "dim",
    "aug",
    "min",
    "maj",
    "N",
]

@dataclass
class ProcessedChordConfig:
    root_dir: str
    n_steps: int = 128
    stride: int = 64
    batch_size: int = 8
    num_workers: int = 0
    augment_train: bool = False
    noise_std: float = 0.01
    gain_min: float = 0.9
    gain_max: float = 1.1
    time_mask_width: int = 8
    freq_mask_width: int = 12
    pitch_shift_bins: int = 0
    use_signal_decay: bool = False
    signal_decay_min: float = 0.4
    signal_decay_max: float = 0.9


class ChordVocab:
    def __init__(self, labels: List[str]):
        uniq = sorted(set(labels))
        if "N" not in uniq:
            uniq.append("N")
        self.idx_to_chord = uniq
        self.chord_to_idx = {c: i for i, c in enumerate(uniq)}

    def encode(self, label: str) -> int:
        return self.chord_to_idx.get(label, self.chord_to_idx["N"])

    def decode(self, idx: int) -> str:
        return self.idx_to_chord[idx]

    @property
    def size(self) -> int:
        return len(self.idx_to_chord)


class FixedChordVocab:
    def __init__(self):
        self.idx_to_chord = TARGET_CLASSES
        self.chord_to_idx = {c: i for i, c in enumerate(self.idx_to_chord)}

    def encode(self, label: str) -> int:
        return self.chord_to_idx[label]

    def decode(self, idx: int) -> str:
        return self.idx_to_chord[idx]

    @property
    def size(self):
        return len(self.idx_to_chord)

def chord_label_to_quality(label: str | None) -> str:
    if label is None:
        return "Others"

    label = str(label).strip()

    if label == "" or label.upper() == "N":
        return "N"

    if ":" not in label:
        if _looks_like_root_label(label):
            return "maj"
        return "Others"

    _, quality = label.split(":", 1)
    quality = quality.strip().replace(" ", "")

    # exact supported classes
    if quality in TARGET_CLASSES:
        return quality

    # optional soft mappings
    if quality == "maj6":
        return "maj"
    if quality == "min6":
        return "min"
    if quality == "maj6/9":
        return "maj9"
    if quality == "min6/9":
        return "min9"
    if quality == "minmaj7":
        return "min7"

    return "Others"


def _looks_like_root_label(label: str) -> bool:
    roots = {
        "C", "C#", "DB", "D", "D#", "EB", "E", "F",
        "F#", "GB", "G", "G#", "AB", "A", "A#", "BB", "B",
    }
    return label.strip().upper() in roots

def load_fold_split(fold_json_path: str):
    with open(fold_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_ids = split_data["train"]
    val_ids = split_data["val"]
    test_ids = split_data["test"]
    return train_ids, val_ids, test_ids


def load_processed_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    cqt = data["cqt"]          # [F, T]
    labels = data["labels"]    # [T]
    sr = int(data["sr"])
    hop_length = int(data["hop_length"])
    frame_rate = float(data["frame_rate"])
    song_id = str(data["song_id"])

    # convert to [T, F]
    x = cqt.T.astype(np.float32)

    labels = labels.tolist()
    labels = [str(lbl) for lbl in labels]

    assert x.shape[0] == len(labels), f"Mismatch: x has {x.shape[0]} frames but labels has {len(labels)}"
    
    return {
        "x": x,                      # [T, F]
        "labels": labels,            # list length T
        "sr": sr,
        "hop_length": hop_length,
        "frame_rate": frame_rate,
        "song_id": song_id,
    }


def make_chord_change_targets(chord_targets: np.ndarray) -> np.ndarray:
    out = np.zeros_like(chord_targets, dtype=np.int64)
    if len(chord_targets) > 1:
        out[1:] = (chord_targets[1:] != chord_targets[:-1]).astype(np.int64)
    out[0] = 0
    return out


def slice_into_windows(
    x: np.ndarray,             # [T, F]
    chord_targets: np.ndarray, # [T]
    chord_change_targets: np.ndarray, # [T]
    n_steps: int,
    stride: int,
):
    total_frames, feat_dim = x.shape
    items = []

    if total_frames <= n_steps:
        pad_len = n_steps - total_frames

        x_pad = np.pad(x, ((0, pad_len), (0, 0)), mode="constant")
        chord_pad = np.pad(chord_targets, (0, pad_len), mode="constant", constant_values=0)
        change_pad = np.pad(chord_change_targets, (0, pad_len), mode="constant", constant_values=0)

        mask = np.zeros(n_steps, dtype=np.float32)
        mask[:total_frames] = 1.0

        items.append({
            "x": x_pad.astype(np.float32),
            "chord_targets": chord_pad.astype(np.int64),
            "chord_change_targets": change_pad.astype(np.int64),
            "mask": mask,
        })
        return items

    for start in range(0, total_frames - n_steps + 1, stride):
        end = start + n_steps
        items.append({
            "x": x[start:end].astype(np.float32),
            "chord_targets": chord_targets[start:end].astype(np.int64),
            "chord_change_targets": chord_change_targets[start:end].astype(np.int64),
            "mask": np.ones(n_steps, dtype=np.float32),
        })

    if (total_frames - n_steps) % stride != 0:
        start = total_frames - n_steps
        end = total_frames
        items.append({
            "x": x[start:end].astype(np.float32),
            "chord_targets": chord_targets[start:end].astype(np.int64),
            "chord_change_targets": chord_change_targets[start:end].astype(np.int64),
            "mask": np.ones(n_steps, dtype=np.float32),
        })

    return items


class ProcessedChordDataset(Dataset):
    def __init__(self, items: List[Dict], augment: bool = False, cfg: ProcessedChordConfig | None = None):
        self.items = items
        self.augment = augment
        self.cfg = cfg

    def __len__(self):
        return len(self.items)

    def _augment_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg is None:
            return x

        if self.cfg.use_signal_decay:
            x = self._apply_signal_decay(x)

        if self.cfg.noise_std > 0:
            x = x + torch.randn_like(x) * self.cfg.noise_std

        if self.cfg.gain_min > 0 and self.cfg.gain_max > 0:
            gain = random.uniform(self.cfg.gain_min, self.cfg.gain_max)
            x = x * gain

        T, F = x.shape

        if self.cfg.time_mask_width > 0 and T > 1:
            width = random.randint(0, min(self.cfg.time_mask_width, T))
            if width > 0:
                start = random.randint(0, T - width)
                x[start:start + width, :] = 0.0

        if self.cfg.freq_mask_width > 0 and F > 1:
            width = random.randint(0, min(self.cfg.freq_mask_width, F))
            if width > 0:
                start = random.randint(0, F - width)
                x[:, start:start + width] = 0.0

        if self.cfg.pitch_shift_bins > 0 and F > 1:
            shift = random.randint(-self.cfg.pitch_shift_bins, self.cfg.pitch_shift_bins)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=1)

        return x

    def _apply_signal_decay(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        if T <= 1:
            return x

        min_gain = min(self.cfg.signal_decay_min, self.cfg.signal_decay_max)
        max_gain = max(self.cfg.signal_decay_min, self.cfg.signal_decay_max)
        end_gain = random.uniform(min_gain, max_gain)
        envelope = torch.linspace(1.0, end_gain, steps=T, dtype=x.dtype, device=x.device).unsqueeze(1)
        return x * envelope

    def __getitem__(self, idx):
        item = self.items[idx]
        x = torch.tensor(item["x"], dtype=torch.float32)
        if self.augment:
            x = self._augment_x(x)

        return {
            "x": x,
            "chord_targets": torch.tensor(item["chord_targets"], dtype=torch.long),
            "chord_change_targets": torch.tensor(item["chord_change_targets"], dtype=torch.long),
            "mask": torch.tensor(item["mask"], dtype=torch.float32),
        }


def build_vocab_from_train_ids(root_dir: str, train_ids: List[str]) -> ChordVocab:
    all_labels = []

    for track_id in train_ids:
        npz_path = os.path.join(root_dir, "processed", f"{track_id}.npz")
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path, allow_pickle=True)
        labels = [str(x) for x in data["labels"].tolist()]
        all_labels.extend(labels)

    return ChordVocab(all_labels)


def build_items_from_ids(root_dir: str, track_ids: List[str], vocab: ChordVocab, cfg: ProcessedChordConfig):
    items = []

    for track_id in track_ids:
        npz_path = os.path.join(root_dir, "processed", f"{track_id}.npz")
        if not os.path.exists(npz_path):
            continue

        sample = load_processed_npz(npz_path)

        x = sample["x"]  # [T, F]
        raw_label_strings = sample["labels"]

        quality_labels = [chord_label_to_quality(lbl) for lbl in raw_label_strings]
        chord_targets = np.array([vocab.encode(lbl) for lbl in quality_labels], dtype=np.int64)
        chord_change_targets = make_chord_change_targets(chord_targets)

        items.extend(
            slice_into_windows(
                x=x,
                chord_targets=chord_targets,
                chord_change_targets=chord_change_targets,
                n_steps=cfg.n_steps,
                stride=cfg.stride,
            )
        )

    return items


def build_processed_loaders(cfg: ProcessedChordConfig, fold_json_path: str):
    train_ids, val_ids, test_ids = load_fold_split(fold_json_path)

    vocab = FixedChordVocab()

    train_items = build_items_from_ids(cfg.root_dir, train_ids, vocab, cfg)
    val_items = build_items_from_ids(cfg.root_dir, val_ids, vocab, cfg)
    test_items = build_items_from_ids(cfg.root_dir, test_ids, vocab, cfg)

    train_dataset = ProcessedChordDataset(train_items, augment=cfg.augment_train, cfg=cfg)
    val_dataset = ProcessedChordDataset(val_items, augment=False, cfg=cfg)
    test_dataset = ProcessedChordDataset(test_items, augment=False, cfg=cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, vocab
