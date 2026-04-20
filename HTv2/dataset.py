import os
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ProcessedChordConfig:
    root_dir: str
    n_steps: int = 128
    stride: int = 64
    batch_size: int = 8
    num_workers: int = 0


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
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "x": torch.tensor(item["x"], dtype=torch.float32),
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
        label_strings = sample["labels"]

        chord_targets = np.array([vocab.encode(lbl) for lbl in label_strings], dtype=np.int64)
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

    vocab = build_vocab_from_train_ids(cfg.root_dir, train_ids)

    train_items = build_items_from_ids(cfg.root_dir, train_ids, vocab, cfg)
    val_items = build_items_from_ids(cfg.root_dir, val_ids, vocab, cfg)
    test_items = build_items_from_ids(cfg.root_dir, test_ids, vocab, cfg)

    train_dataset = ProcessedChordDataset(train_items)
    val_dataset = ProcessedChordDataset(val_items)
    test_dataset = ProcessedChordDataset(test_items)

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