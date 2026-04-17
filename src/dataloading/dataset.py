"""
dataset.py - BeatmapDataset v2
Tre-kanal input: mel-spektrogram + onset envelope + tempogram
Target: [x, y, circle, slider_prob, slider_end_x, slider_end_y, spinner]  (7 verdier)
"""

import os
import json
import random
import warnings
import logging

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
logging.getLogger("librosa").setLevel(logging.ERROR)

# ─────────────────────────────────────────────
# KONSTANTER
# ─────────────────────────────────────────────
SR          = 22050
HOP_LENGTH  = 512
N_MELS      = 128
N_FFT       = 2048
PLAYFIELD_X = 512.0
PLAYFIELD_Y = 384.0

# Output-indekser
N_OUTPUTS        = 7
IDX_X            = 0
IDX_Y            = 1
IDX_CIRCLE       = 2
IDX_SLIDER_PROB  = 3
IDX_SLIDER_END_X = 4
IDX_SLIDER_END_Y = 5
IDX_SPINNER      = 6


# ─────────────────────────────────────────────
# AUDIO FEATURES
# ─────────────────────────────────────────────
def load_audio_features(audio_path: str) -> np.ndarray:
    """
    Returnerer (3, N_MELS, T):
      Kanal 0: Mel-spektrogram (normalisert)
      Kanal 1: Onset envelope  (broadcastet til N_MELS rader)
      Kanal 2: Tempogram       (resamplet til N_MELS rader)
    """
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    # ── Kanal 0: Mel-spektrogram ──
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                           hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
    T    = S_db.shape[1]

    # ── Kanal 1: Onset envelope ──
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset = onset[:T]
    if len(onset) < T:
        onset = np.pad(onset, (0, T - len(onset)))
    onset = (onset - onset.mean()) / (onset.std() + 1e-8)
    onset_2d = np.tile(onset[np.newaxis, :], (N_MELS, 1))

    # ── Kanal 2: Tempogram ──
    try:
        tempo_raw = librosa.feature.tempogram(y=y, sr=sr, hop_length=HOP_LENGTH,
                                               win_length=384)
        # Juster tidsaksen
        if tempo_raw.shape[1] != T:
            indices  = np.round(np.linspace(0, tempo_raw.shape[1]-1, T)).astype(int)
            tempo_raw = tempo_raw[:, indices]
        # Resample frekvensakse fra (192, T) til (128, T)
        freq_indices = np.round(np.linspace(0, tempo_raw.shape[0]-1, N_MELS)).astype(int)
        tempo_2d     = tempo_raw[freq_indices, :]
        tempo_2d     = (tempo_2d - tempo_2d.mean()) / (tempo_2d.std() + 1e-8)
    except Exception:
        tempo_2d = onset_2d.copy()

    features = np.stack([S_db, onset_2d, tempo_2d], axis=0).astype(np.float32)
    return features  # (3, 128, T)


# ─────────────────────────────────────────────
# TARGET PARSER
# ─────────────────────────────────────────────
def parse_hit_objects(beatmap: dict, T: int) -> np.ndarray:
    """
    Konverterer hit objects til (N_OUTPUTS, T) target-matrise.

    osu! type bits:
      bit 0 (1) = circle
      bit 1 (2) = slider
      bit 3 (8) = spinner
    """
    target      = np.zeros((N_OUTPUTS, T), dtype=np.float32)
    hit_objects = beatmap.get("hit_objects", [])

    for obj in hit_objects:
        try:
            frame = int(float(obj["time"]) / 1000.0 * SR / HOP_LENGTH)
            if frame < 0 or frame >= T:
                continue

            x_norm = np.clip(float(obj["x"]) / PLAYFIELD_X, 0.0, 1.0)
            y_norm = np.clip(float(obj["y"]) / PLAYFIELD_Y, 0.0, 1.0)

            target[IDX_X, frame] = x_norm
            target[IDX_Y, frame] = y_norm

            obj_type = int(obj.get("type", 1))

            if obj_type & 8:  # Spinner
                target[IDX_SPINNER, frame] = 1.0

            elif obj_type & 2:  # Slider
                target[IDX_SLIDER_PROB, frame] = 1.0

                end_x = obj.get("end_x")
                end_y = obj.get("end_y")

                if end_x is not None and end_y is not None:
                    target[IDX_SLIDER_END_X, frame] = np.clip(float(end_x) / PLAYFIELD_X, 0.0, 1.0)
                    target[IDX_SLIDER_END_Y, frame] = np.clip(float(end_y) / PLAYFIELD_Y, 0.0, 1.0)
                else:
                    # Estimer slutt basert på retning fra senter
                    dx   = x_norm - 0.5
                    dy   = y_norm - 0.5
                    mag  = max(0.05, np.sqrt(dx**2 + dy**2))
                    target[IDX_SLIDER_END_X, frame] = np.clip(x_norm + (dx/mag) * 0.2, 0.0, 1.0)
                    target[IDX_SLIDER_END_Y, frame] = np.clip(y_norm + (dy/mag) * 0.2, 0.0, 1.0)

            else:  # Circle
                target[IDX_CIRCLE, frame] = 1.0

        except (KeyError, ValueError, TypeError):
            continue

    return target


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class BeatmapDataset(Dataset):
    def __init__(
        self,
        json_path:      str,
        segment_length: int  = 3000,
        augment:        bool = True,
        min_notes:      int  = 30,
    ):
        self.segment_length = segment_length
        self.augment        = augment

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples   = []
        skipped        = 0
        total_sliders  = 0
        total_circles  = 0

        for song in raw:
            # Finn audio-fil
            audio_path = song.get("audio_path", "")
            if not audio_path or not os.path.exists(audio_path):
                folder = song.get("folder_path", "")
                audio_path = ""
                for ext in ["audio.mp3", "audio.ogg", "audio.wav"]:
                    c = os.path.join(folder, ext)
                    if os.path.exists(c):
                        audio_path = c
                        break

            if not audio_path:
                skipped += 1
                continue

            for bm in song.get("beatmaps", []):
                hit_objects = bm.get("hit_objects", [])
                if len(hit_objects) < min_notes:
                    skipped += 1
                    continue

                n_sliders = sum(1 for o in hit_objects if int(o.get("type", 1)) & 2)
                n_circles = sum(1 for o in hit_objects
                                if (int(o.get("type", 1)) & 1)
                                and not (int(o.get("type", 1)) & 2))
                total_sliders += n_sliders
                total_circles += n_circles

                diff = bm.get("difficulty", {})
                self.samples.append({
                    "audio_path": audio_path,
                    "beatmap":    bm,
                    "bpm":        float(bm.get("bpm", 120) or 120),
                    "ar":         float(diff.get("ar", 9)  or 9),
                    "od":         float(diff.get("od", 8)  or 8),
                    "cs":         float(diff.get("cs", 4)  or 4),
                    "hp":         float(diff.get("hp", 5)  or 5),
                })

        slider_pct = 100 * total_sliders / max(total_sliders + total_circles, 1)
        print(f"Dataset lastet: {len(self.samples)} beatmaps ({skipped} hoppet over)")
        print(f"  Circles: {total_circles:,} | Sliders: {total_sliders:,} ({slider_pct:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        try:
            features = load_audio_features(s["audio_path"])  # (3, 128, T)
            T        = features.shape[2]

            # Pad hvis for kort
            if T < self.segment_length:
                pad      = self.segment_length - T
                features = np.pad(features, ((0,0),(0,0),(0,pad)))
                T        = features.shape[2]

            # Random segment
            start    = random.randint(0, T - self.segment_length)
            end      = start + self.segment_length
            features = features[:, :, start:end]

            # Target
            target = parse_hit_objects(s["beatmap"], T)
            target = target[:, start:end]

            # Augmentering: liten pitch-shift via mel-roll
            if self.augment and random.random() < 0.3:
                shift      = random.randint(-2, 2)
                features[0] = np.roll(features[0], shift, axis=0)

            # Difficulty-vektor
            diff = np.array([
                s["ar"]  / 10.0,
                s["od"]  / 10.0,
                s["cs"]  / 7.0,
                s["hp"]  / 10.0,
            ], dtype=np.float32)
            bpm_v = np.array([s["bpm"] / 200.0], dtype=np.float32)

            return {
                "input":   torch.tensor(features, dtype=torch.float32),
                "target":  torch.tensor(target,   dtype=torch.float32),
                "diff":    torch.tensor(diff,     dtype=torch.float32),
                "bpm":     torch.tensor(bpm_v,    dtype=torch.float32),
                "title":   s["beatmap"].get("title",   ""),
                "version": s["beatmap"].get("version", ""),
            }

        except Exception as e:
            print(f"\nFeil på {s['audio_path']}: {str(e)[:80]}")
            return self.__getitem__(random.randint(0, len(self) - 1))


# ─────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    min_t = min(b["input"].shape[2] for b in batch)

    return {
        "input":  torch.stack([b["input"][:, :, :min_t]  for b in batch]),
        "target": torch.stack([b["target"][:, :min_t]    for b in batch]),
        "diff":   torch.stack([b["diff"]                 for b in batch]),
        "bpm":    torch.stack([b["bpm"]                  for b in batch]),
    }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds     = BeatmapDataset("hard_data.json", segment_length=3000)
    sample = ds[0]

    print(f"\nInput shape:  {sample['input'].shape}")   # (3, 128, 3000)
    print(f"Target shape: {sample['target'].shape}")   # (7, 3000)
    print(f"Diff:         {sample['diff']}")
    print(f"BPM:          {sample['bpm'].item():.3f}")
    print(f"Tittel:       {sample['title']} [{sample['version']}]")

    t = sample["target"]
    print(f"\nNote-fordeling i segment:")
    print(f"  Circles:  {t[IDX_CIRCLE].sum():.0f}")
    print(f"  Sliders:  {t[IDX_SLIDER_PROB].sum():.0f}")
    print(f"  Spinners: {t[IDX_SPINNER].sum():.0f}")

    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
    batch  = next(iter(loader))

    print(f"\nBatch input:  {batch['input'].shape}")   # (4, 3, 128, 3000)
    print(f"Batch target: {batch['target'].shape}")   # (4, 7, 3000)
    print(f"\nDataset klar!")