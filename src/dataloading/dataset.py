"""
dataset.py - PyTorch Dataset for osu! Beatmap Generator
Laster hard_data.json og returnerer spektrogram + note-matrise
"""

import json
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ─────────────────────────────────────────────
# KONSTANTER
# ─────────────────────────────────────────────
HOP_LENGTH  = 512
N_MELS      = 128
SR          = 22050
PLAYFIELD_X = 512
PLAYFIELD_Y = 384


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def ms_to_frame(ms, sr=SR, hop_length=HOP_LENGTH):
    return int((ms / 1000) * sr / hop_length)


def parse_bpm(timing_points):
    for tp in timing_points:
        try:
            if int(tp[6]) == 1:
                return round(60000 / float(tp[1]), 1)
        except:
            continue
    return 120.0  # fallback


def build_note_matrix(hit_objects, n_frames):
    """Lager (5 x n_frames) matrise med noter"""
    matrix = np.zeros((5, n_frames), dtype=np.float32)

    for note in hit_objects:
        try:
            x        = int(note[0]) / PLAYFIELD_X
            y        = int(note[1]) / PLAYFIELD_Y
            ts_ms    = int(note[2])
            type_raw = int(note[3])

            circle  = 1.0 if type_raw in [1, 5]  else 0.0
            slider  = 1.0 if type_raw in [2, 6]  else 0.0
            spinner = 1.0 if type_raw == 12       else 0.0

            col = ms_to_frame(ts_ms)
            if 0 <= col < n_frames:
                matrix[0, col] = x
                matrix[1, col] = y
                matrix[2, col] = circle
                matrix[3, col] = slider
                matrix[4, col] = spinner

        except (ValueError, IndexError):
            continue

    return matrix


def load_spectrogram(audio_path):
    """Laster MP3 og returnerer normalisert mel-spektrogram"""
    y, sr = librosa.load(audio_path, sr=SR)
    S     = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_db  = librosa.power_to_db(S, ref=np.max)
    S_db  = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
    return S_db


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class BeatmapDataset(Dataset):
    def __init__(self, json_path, segment_length=None, cache=False):
        """
        Args:
            json_path:       Sti til hard_data.json
            segment_length:  Antall frames per segment (None = hele sangen)
            cache:           Cache spektrogrammer i minnet (raskere men mer RAM)
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Flat liste av alle beatmaps
        self.beatmaps = []
        for sang in data:
            for bm in sang["beatmaps"]:
                self.beatmaps.append({
                    "audio_path":   bm["audio_path"],
                    "hit_objects":  bm["hit_objects"],
                    "timing_points": bm["timing_points"],
                    "difficulty":   bm["difficulty"],
                    "title":        bm["metadata"].get("title", ""),
                    "version":      bm["metadata"].get("version", ""),
                })

        self.segment_length = segment_length
        self.cache          = cache
        self._cache         = {}

        print(f"Dataset lastet: {len(self.beatmaps)} beatmaps")

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        bm = self.beatmaps[idx]

        # Hent eller cache spektrogram
        if self.cache and idx in self._cache:
            S_db = self._cache[idx]
        else:
            try:
                S_db = load_spectrogram(bm["audio_path"])
            except Exception as e:
                print(f"Feil på {bm['audio_path']}: {e}")
                return None

            if self.cache:
                self._cache[idx] = S_db

        n_frames     = S_db.shape[1]
        note_matrix  = build_note_matrix(bm["hit_objects"], n_frames)

        # Valgfri segmentering
        if self.segment_length and n_frames > self.segment_length:
            start   = np.random.randint(0, n_frames - self.segment_length)
            S_db    = S_db[:, start:start + self.segment_length]
            note_matrix = note_matrix[:, start:start + self.segment_length]

        # Difficulty kondisjonering
        diff = bm["difficulty"]
        diff_vector = np.array([
            diff.get("ar", 8) / 10,
            diff.get("od", 8) / 10,
            diff.get("cs", 4) / 7,
            diff.get("hp", 5) / 10,
        ], dtype=np.float32)

        # BPM normalisert
        bpm = parse_bpm(bm["timing_points"])
        bpm_norm = np.array([bpm / 200.0], dtype=np.float32)

        input_tensor  = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)  # (1, 128, T)
        target_tensor = torch.tensor(note_matrix, dtype=torch.float32)         # (5, T)
        diff_tensor   = torch.tensor(diff_vector, dtype=torch.float32)         # (4,)
        bpm_tensor    = torch.tensor(bpm_norm, dtype=torch.float32)            # (1,)

        return {
            "input":  input_tensor,   # Spektrogram
            "target": target_tensor,  # Note-matrise
            "diff":   diff_tensor,    # AR, OD, CS, HP
            "bpm":    bpm_tensor,     # BPM normalisert
            "title":  bm["title"],
            "version": bm["version"],
        }


# ─────────────────────────────────────────────
# COLLATE - håndterer ulik sanglengde i batch
# ─────────────────────────────────────────────
def collate_fn(batch):
    """Trimmer alle sanger i batchen til korteste lengde"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    min_len = min(b["input"].shape[2] for b in batch)

    inputs  = torch.stack([b["input"][:, :, :min_len]  for b in batch])
    targets = torch.stack([b["target"][:, :min_len]    for b in batch])
    diffs   = torch.stack([b["diff"]                   for b in batch])
    bpms    = torch.stack([b["bpm"]                    for b in batch])

    return {
        "input":  inputs,   # (B, 1, 128, T)
        "target": targets,  # (B, 5, T)
        "diff":   diffs,    # (B, 4)
        "bpm":    bpms,     # (B, 1)
    }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    dataset = BeatmapDataset("hard_data.json")

    # Test første element
    sample = dataset[0]
    print(f"\nTittel:  {sample['title']} [{sample['version']}]")
    print(f"Input:   {sample['input'].shape}")
    print(f"Target:  {sample['target'].shape}")
    print(f"Diff:    {sample['diff']}")
    print(f"BPM:     {sample['bpm'].item() * 200:.1f}")

    # Test DataLoader med batch
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    batch = next(iter(loader))
    if batch:
        print(f"\nBatch input shape:  {batch['input'].shape}")
        print(f"Batch target shape: {batch['target'].shape}")
        print(f"Batch diff shape:   {batch['diff'].shape}")