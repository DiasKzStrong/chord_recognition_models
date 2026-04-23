import os
import json
from dataclasses import dataclass
from typing import Dict, List
import random
import re

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
    window_mode: str = "sliding"
    label_mode: str = "quality27"
    augment_train: bool = False
    noise_std: float = 0.01
    gain_min: float = 0.9
    gain_max: float = 1.1
    time_mask_width: int = 8
    freq_mask_width: int = 12
    pitch_shift_bins: int = 0
    pitch_shift_semitones: int = 0
    use_signal_decay: bool = False
    signal_decay_min: float = 0.4
    signal_decay_max: float = 0.9


class ChordVocab:
    def __init__(self, labels: List[str], label_mode: str = "full_chord"):
        uniq = sorted(set(labels))
        if "N" not in uniq:
            uniq.append("N")
        self.label_mode = label_mode
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
        self.label_mode = "quality27"
        self.idx_to_chord = TARGET_CLASSES
        self.chord_to_idx = {c: i for i, c in enumerate(self.idx_to_chord)}

    def encode(self, label: str) -> int:
        return self.chord_to_idx[label]

    def decode(self, idx: int) -> str:
        return self.idx_to_chord[idx]

    @property
    def size(self):
        return len(self.idx_to_chord)


ROOT_TO_CANONICAL = {
    "C": "C",
    "B#": "C",
    "C#": "C#",
    "DB": "C#",
    "D": "D",
    "D#": "D#",
    "EB": "D#",
    "E": "E",
    "FB": "E",
    "E#": "F",
    "F": "F",
    "F#": "F#",
    "GB": "F#",
    "G": "G",
    "G#": "G#",
    "AB": "G#",
    "A": "A",
    "A#": "A#",
    "BB": "A#",
    "B": "B",
    "CB": "B",
}

CANONICAL_ROOTS = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]
FULL_CHORD_QUALITIES = [q for q in TARGET_CLASSES if q not in {"Others", "N"}]
TRIAD_TYPES = ["maj", "min", "dim", "aug", "sus4", "sus2"]
STRUCTURED_COMPONENT_NAMES = [
    "root_triad",
    "bass",
    "seventh",
    "ninth",
    "eleventh",
    "thirteenth",
]
STRUCTURED_COMPONENT_LABELS = {
    "root_triad": ["N"] + [
        f"{root}:{triad}"
        for root in CANONICAL_ROOTS
        for triad in TRIAD_TYPES
    ],
    "bass": ["N", "root", "/2", "/b3", "/3", "/4", "/5", "/6", "/b7"],
    "seventh": ["N", "none", "+7", "+b7", "+bb7"],
    "ninth": ["N", "none", "+9", "+#9", "+b9"],
    "eleventh": ["N", "none", "+11", "+#11"],
    "thirteenth": ["N", "none", "+13", "+b13"],
}


def canonicalize_root(root: str) -> str | None:
    return ROOT_TO_CANONICAL.get(root.strip().upper())


def normalize_quality(quality: str | None) -> str:
    if quality is None:
        return "maj"

    quality = str(quality).strip().replace(" ", "")
    quality = quality.replace("*", "")
    if quality == "":
        return "maj"
    if quality in {"1", "1/1"}:
        return "maj"
    if quality.startswith("5"):
        return "maj"
    if quality in TARGET_CLASSES and quality not in {"N", "Others"}:
        return quality

    base = re.sub(r"\([^)]*\)", "", quality)
    if "/" in base:
        base_no_bass = base.split("/", 1)[0]
    else:
        base_no_bass = base

    if quality.startswith("maj(9)") or quality.startswith("maj9"):
        return "maj9"
    if quality.startswith("min(9)") or quality.startswith("min9"):
        return "min9"
    if quality.startswith("7(") or quality.startswith("7/"):
        return "9" if "9" in quality else "7"
    if quality.startswith("maj7"):
        return "maj7"
    if quality.startswith("min7"):
        return "min7"
    if quality.startswith("sus4(b7") or quality.startswith("sus4/b7"):
        return "sus4(b7)"
    if quality.startswith("sus4"):
        return "sus4"
    if quality.startswith("sus2"):
        return "sus2"
    if quality.startswith("dim7"):
        return "dim7"
    if quality.startswith("hdim7"):
        return "hdim7"
    if quality.startswith("dim/b7"):
        return "dim7"
    if quality.startswith("dim"):
        return "dim"
    if quality.startswith("aug"):
        return "aug"
    if quality.startswith("min11"):
        return "min9"
    if quality.startswith("min6"):
        return "min"
    if quality.startswith("maj6"):
        return "maj"
    if quality.startswith("maj"):
        mapped_bass = _map_bass_marker(base)
        return f"maj{mapped_bass}" if mapped_bass else "maj"
    if quality.startswith("min"):
        mapped_bass = _map_bass_marker(base)
        return f"min{mapped_bass}" if mapped_bass else "min"
    if base_no_bass in TARGET_CLASSES and base_no_bass not in {"N", "Others"}:
        return base_no_bass

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


def _map_bass_marker(quality: str) -> str | None:
    if "/" not in quality:
        return None
    bass = quality.split("/", 1)[1]
    if bass in {"2", "b3", "3", "5", "b7"}:
        return f"/{bass}"
    return None


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
    return normalize_quality(quality)


def chord_label_to_full_chord(label: str | None) -> str:
    if label is None:
        return "N"

    label = str(label).strip()
    if label == "" or label.upper() == "N":
        return "N"

    if ":" not in label:
        if "/" in label:
            root_raw, bass_raw = label.split("/", 1)
            root = canonicalize_root(root_raw)
            if root is not None and bass_raw in {"2", "b3", "3", "5", "b7"}:
                return f"{root}:maj/{bass_raw}"
        root = canonicalize_root(label)
        return f"{root}:maj" if root is not None else "N"

    root_raw, quality_raw = label.split(":", 1)
    root = canonicalize_root(root_raw)
    if root is None:
        return "N"

    quality = normalize_quality(quality_raw)
    if quality in {"N", "Others"}:
        return "N"
    return f"{root}:{quality}"


def full_chord_to_components(label: str | None) -> List[str]:
    label = chord_label_to_full_chord(label)
    if label == "N":
        return ["N", "N", "N", "N", "N", "N"]

    root, quality = label.split(":", 1)

    triad = "maj"
    bass = "root"
    seventh = "none"
    ninth = "none"
    eleventh = "none"
    thirteenth = "none"

    if quality.startswith("min"):
        triad = "min"
    elif quality.startswith("dim") or quality == "hdim7":
        triad = "dim"
    elif quality == "aug":
        triad = "aug"
    elif quality.startswith("sus4"):
        triad = "sus4"
    elif quality.startswith("sus2"):
        triad = "sus2"

    bass_markers = {
        "maj/2": "/2",
        "min/2": "/2",
        "min/b3": "/b3",
        "maj/3": "/3",
        "maj/5": "/5",
        "min/5": "/5",
        "maj/b7": "/b7",
        "min/b7": "/b7",
    }
    bass = bass_markers.get(quality, "root")

    if quality == "maj7":
        seventh = "+7"
    elif quality in {"7", "9", "11", "13", "min7", "min9", "sus4(b7)", "hdim7"}:
        seventh = "+b7"
    elif quality == "dim7":
        seventh = "+bb7"

    if quality in {"9", "maj9", "min9", "11", "13"}:
        ninth = "+9"
    if quality in {"11", "13"}:
        eleventh = "+11"
    if quality == "13":
        thirteenth = "+13"

    return [
        f"{root}:{triad}",
        bass,
        seventh,
        ninth,
        eleventh,
        thirteenth,
    ]


def build_structured_component_vocabs():
    return {
        name: {label: idx for idx, label in enumerate(labels)}
        for name, labels in STRUCTURED_COMPONENT_LABELS.items()
    }


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
    chord_label_strings: List[str],
    component_targets: np.ndarray | None,
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
        if component_targets is not None:
            component_pad = np.pad(
                component_targets,
                ((0, pad_len), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            component_pad = None

        mask = np.zeros(n_steps, dtype=np.float32)
        mask[:total_frames] = 1.0

        item = {
            "x": x_pad.astype(np.float32),
            "chord_targets": chord_pad.astype(np.int64),
            "chord_change_targets": change_pad.astype(np.int64),
            "chord_label_strings": chord_label_strings + ["N"] * pad_len,
            "mask": mask,
        }
        if component_pad is not None:
            item["component_targets"] = component_pad.astype(np.int64)
        items.append(item)
        return items

    for start in range(0, total_frames - n_steps + 1, stride):
        end = start + n_steps
        item = {
            "x": x[start:end].astype(np.float32),
            "chord_targets": chord_targets[start:end].astype(np.int64),
            "chord_change_targets": chord_change_targets[start:end].astype(np.int64),
            "chord_label_strings": chord_label_strings[start:end],
            "mask": np.ones(n_steps, dtype=np.float32),
        }
        if component_targets is not None:
            item["component_targets"] = component_targets[start:end].astype(np.int64)
        items.append(item)

    if (total_frames - n_steps) % stride != 0:
        start = total_frames - n_steps
        end = total_frames
        item = {
            "x": x[start:end].astype(np.float32),
            "chord_targets": chord_targets[start:end].astype(np.int64),
            "chord_change_targets": chord_change_targets[start:end].astype(np.int64),
            "chord_label_strings": chord_label_strings[start:end],
            "mask": np.ones(n_steps, dtype=np.float32),
        }
        if component_targets is not None:
            item["component_targets"] = component_targets[start:end].astype(np.int64)
        items.append(item)

    return items


def make_song_item(
    x: np.ndarray,
    chord_targets: np.ndarray,
    chord_change_targets: np.ndarray,
    chord_label_strings: List[str],
    component_targets: np.ndarray | None,
):
    item = {
        "x": x.astype(np.float32),
        "chord_targets": chord_targets.astype(np.int64),
        "chord_change_targets": chord_change_targets.astype(np.int64),
        "chord_label_strings": chord_label_strings,
    }
    if component_targets is not None:
        item["component_targets"] = component_targets.astype(np.int64)
    return item


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

        if self.cfg.pitch_shift_semitones > 0 and F > 1:
            semitones = random.randint(-self.cfg.pitch_shift_semitones, self.cfg.pitch_shift_semitones)
            shift_bins = semitones * 3
            if shift_bins != 0:
                x = torch.roll(x, shifts=shift_bins, dims=1)

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

        sample = {
            "x": x,
            "chord_targets": torch.tensor(item["chord_targets"], dtype=torch.long),
            "chord_change_targets": torch.tensor(item["chord_change_targets"], dtype=torch.long),
            "mask": torch.tensor(item["mask"], dtype=torch.float32),
        }
        if "component_targets" in item:
            sample["component_targets"] = torch.tensor(item["component_targets"], dtype=torch.long)
        return sample


class RandomSongSegmentDataset(Dataset):
    def __init__(self, songs: List[Dict], cfg: ProcessedChordConfig, augment: bool = False):
        self.songs = songs
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.songs)

    def _slice_song(self, item: Dict):
        x = item["x"]
        chord_targets = item["chord_targets"]
        chord_change_targets = item["chord_change_targets"]
        component_targets = item.get("component_targets")
        total_frames = x.shape[0]
        n_steps = self.cfg.n_steps

        if total_frames <= n_steps:
            pad_len = n_steps - total_frames
            x_out = np.pad(x, ((0, pad_len), (0, 0)), mode="constant")
            chord_out = np.pad(chord_targets, (0, pad_len), mode="constant", constant_values=0)
            change_out = np.pad(chord_change_targets, (0, pad_len), mode="constant", constant_values=0)
            if component_targets is not None:
                component_out = np.pad(
                    component_targets,
                    ((0, pad_len), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                component_out = None
            mask = np.zeros(n_steps, dtype=np.float32)
            mask[:total_frames] = 1.0
            label_strings = item["chord_label_strings"][:total_frames] + ["N"] * pad_len
        else:
            start = random.randint(0, total_frames - n_steps)
            end = start + n_steps
            x_out = x[start:end]
            chord_out = chord_targets[start:end]
            change_out = chord_change_targets[start:end]
            component_out = component_targets[start:end] if component_targets is not None else None
            mask = np.ones(n_steps, dtype=np.float32)
            label_strings = item["chord_label_strings"][start:end]

        out = {
            "x": x_out.astype(np.float32),
            "chord_targets": chord_out.astype(np.int64),
            "chord_change_targets": change_out.astype(np.int64),
            "chord_label_strings": label_strings,
            "mask": mask,
        }
        if component_out is not None:
            out["component_targets"] = component_out.astype(np.int64)
        return out

    def __getitem__(self, idx):
        item = self._slice_song(self.songs[idx])
        sample = ProcessedChordDataset([item], augment=self.augment, cfg=self.cfg)[0]
        return sample


def label_to_target(label: str, label_mode: str) -> str:
    if label_mode == "quality27":
        return chord_label_to_quality(label)
    if label_mode in {"full_chord", "structured_full_chord"}:
        return chord_label_to_full_chord(label)
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def build_vocab_from_train_ids(root_dir: str, train_ids: List[str], label_mode: str) -> ChordVocab:
    all_labels = []

    for track_id in train_ids:
        npz_path = os.path.join(root_dir, "processed", f"{track_id}.npz")
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path, allow_pickle=True)
        labels = [str(x) for x in data["labels"].tolist()]
        all_labels.extend(label_to_target(lbl, label_mode) for lbl in labels)

    return ChordVocab(all_labels, label_mode=label_mode)


def attach_structured_metadata(vocab: ChordVocab) -> ChordVocab:
    component_vocabs = build_structured_component_vocabs()
    component_ids = []
    for chord in vocab.idx_to_chord:
        components = full_chord_to_components(chord)
        component_ids.append([
            component_vocabs[name][component]
            for name, component in zip(STRUCTURED_COMPONENT_NAMES, components)
        ])

    vocab.component_names = STRUCTURED_COMPONENT_NAMES
    vocab.component_labels = STRUCTURED_COMPONENT_LABELS
    vocab.component_to_idx = component_vocabs
    vocab.chord_component_ids = np.array(component_ids, dtype=np.int64)
    return vocab


def build_full_chord_vocab(label_mode: str = "full_chord") -> ChordVocab:
    labels = [
        f"{root}:{quality}"
        for root in CANONICAL_ROOTS
        for quality in FULL_CHORD_QUALITIES
    ]
    labels.append("N")
    vocab = ChordVocab(labels, label_mode=label_mode)
    return attach_structured_metadata(vocab)


def encode_component_targets(labels: List[str], vocab: ChordVocab) -> np.ndarray:
    rows = []
    for label in labels:
        components = full_chord_to_components(label)
        rows.append([
            vocab.component_to_idx[name][component]
            for name, component in zip(vocab.component_names, components)
        ])
    return np.array(rows, dtype=np.int64)


def build_items_from_ids(
    root_dir: str,
    track_ids: List[str],
    vocab: ChordVocab,
    cfg: ProcessedChordConfig,
    window_mode: str | None = None,
):
    items = []
    mode = window_mode or cfg.window_mode

    for track_id in track_ids:
        npz_path = os.path.join(root_dir, "processed", f"{track_id}.npz")
        if not os.path.exists(npz_path):
            continue

        sample = load_processed_npz(npz_path)

        x = sample["x"]  # [T, F]
        raw_label_strings = sample["labels"]

        target_labels = [label_to_target(lbl, cfg.label_mode) for lbl in raw_label_strings]
        chord_targets = np.array([vocab.encode(lbl) for lbl in target_labels], dtype=np.int64)
        chord_change_targets = make_chord_change_targets(chord_targets)
        component_targets = None
        if cfg.label_mode == "structured_full_chord":
            component_targets = encode_component_targets(target_labels, vocab)

        if mode == "random_song":
            items.append(
                make_song_item(
                    x=x,
                    chord_targets=chord_targets,
                    chord_change_targets=chord_change_targets,
                    chord_label_strings=target_labels,
                    component_targets=component_targets,
                )
            )
        elif mode == "sliding":
            items.extend(
                slice_into_windows(
                    x=x,
                    chord_targets=chord_targets,
                    chord_change_targets=chord_change_targets,
                    chord_label_strings=target_labels,
                    component_targets=component_targets,
                    n_steps=cfg.n_steps,
                    stride=cfg.stride,
                )
            )
        else:
            raise ValueError(f"Unsupported window_mode: {mode}")

    return items


def build_processed_loaders(cfg: ProcessedChordConfig, fold_json_path: str):
    train_ids, val_ids, test_ids = load_fold_split(fold_json_path)

    if cfg.label_mode == "quality27":
        vocab = FixedChordVocab()
    elif cfg.label_mode in {"full_chord", "structured_full_chord"}:
        vocab = build_full_chord_vocab(label_mode=cfg.label_mode)
    else:
        raise ValueError(f"Unsupported label_mode: {cfg.label_mode}")

    train_items = build_items_from_ids(cfg.root_dir, train_ids, vocab, cfg, window_mode=cfg.window_mode)
    val_items = build_items_from_ids(cfg.root_dir, val_ids, vocab, cfg, window_mode="sliding")
    test_items = build_items_from_ids(cfg.root_dir, test_ids, vocab, cfg, window_mode="sliding")

    if cfg.window_mode == "random_song":
        train_dataset = RandomSongSegmentDataset(train_items, cfg=cfg, augment=cfg.augment_train)
        val_dataset = ProcessedChordDataset(val_items, augment=False, cfg=cfg)
        test_dataset = ProcessedChordDataset(test_items, augment=False, cfg=cfg)
    else:
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
