"""
generate.py - Generer osu! beatmap fra en MP3-fil
Bruk: python generate.py --audio sang.mp3 --output generated.osu --model checkpoints/beatmap_model.pth
"""

import os
import sys
import argparse
import numpy as np
import librosa
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import BeatmapGenerator

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
def detect_bpm(audio_path):
    """Detekter BPM fra MP3"""
    y, sr = librosa.load(audio_path, sr=SR)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    print(f"Detektert BPM: {bpm:.1f}")
    return bpm


def load_spectrogram(audio_path):
    """Last inn MP3 og lag normalisert mel-spektrogram"""
    y, sr = librosa.load(audio_path, sr=SR)
    S     = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_db  = librosa.power_to_db(S, ref=np.max)
    S_db  = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
    return S_db, sr


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ─────────────────────────────────────────────
# GENERERING
# ─────────────────────────────────────────────
def generer_map(
    model,
    audio_path,
    output_path,
    device,
    ar          = 9.0,
    od          = 8.0,
    cs          = 4.0,
    hp          = 5.0,
    threshold   = 0.3,
    title       = "AI Generated",
    artist      = "Unknown",
    version     = "AI Hard",
):
    print(f"\nLaster inn: {audio_path}")

    # Spektrogram
    S_db, sr = load_spectrogram(audio_path)
    bpm      = detect_bpm(audio_path)

    # Tensorer
    input_tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0).float().contiguous().to(device)
    diff_tensor  = torch.tensor([[ar/10, od/10, cs/7, hp/10]], dtype=torch.float32).to(device)
    bpm_tensor   = torch.tensor([[bpm/200]], dtype=torch.float32).to(device)

    print(f"Spektrogram shape: {input_tensor.shape}")
    print(f"Difficulty: AR{ar} OD{od} CS{cs} HP{hp}")

    # Generer
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor, diff_tensor, bpm_tensor)
        pred = torch.sigmoid(pred)

    pred_np = pred.squeeze(0).cpu().numpy()  # (T, 5)

    print(f"\nPrediksjoner:")
    print(f"  Max circle: {pred_np[:, 2].max():.4f}")
    print(f"  Max slider: {pred_np[:, 3].max():.4f}")
    print(f"  Threshold:  {threshold}")

    # Ekstraher noter
    noter = []
    for kolonne in range(pred_np.shape[0]):
        x_raw   = pred_np[kolonne, 0]
        y_raw   = pred_np[kolonne, 1]
        circle  = pred_np[kolonne, 2]
        slider  = pred_np[kolonne, 3]

        if circle > threshold or slider > threshold:
            timestamp_ms = int(kolonne * HOP_LENGTH / sr * 1000)

            # Sigmoid på x og y for å holde innenfor playfield
            x     = float(sigmoid(x_raw))
            y_pos = float(sigmoid(y_raw))

            noter.append({
                "x":         max(0, min(511, int(x * PLAYFIELD_X))),
                "y":         max(0, min(383, int(y_pos * PLAYFIELD_Y))),
                "timestamp": timestamp_ms,
                "type":      1 if circle >= slider else 2,
            })

    print(f"  Generert:   {len(noter)} noter")

    if len(noter) == 0:
        print("\nINGEN NOTER GENERERT – prøv lavere threshold (f.eks. --threshold 0.1)")
        return []

    # Fjern duplikater (noter for tett i tid)
    min_gap_ms = int(60000 / bpm / 4)  # Minst 1/4 beat mellom noter
    filtrerte  = [noter[0]]
    for note in noter[1:]:
        if note["timestamp"] - filtrerte[-1]["timestamp"] >= min_gap_ms:
            filtrerte.append(note)

    print(f"  Etter filtrering: {len(filtrerte)} noter (min gap: {min_gap_ms}ms)")

    # BPM til ms_per_beat
    ms_per_beat = 60000 / bpm

    # Skriv .osu fil
    audio_filename = Path(audio_path).name
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("osu file format v14\n\n")

        f.write("[General]\n")
        f.write(f"AudioFilename: {audio_filename}\n")
        f.write("AudioLeadIn: 0\n")
        f.write("PreviewTime: -1\n")
        f.write("Countdown: 0\n")
        f.write("Mode: 0\n\n")

        f.write("[Metadata]\n")
        f.write(f"Title: {title}\n")
        f.write(f"Artist: {artist}\n")
        f.write("Creator: AI\n")
        f.write(f"Version: {version}\n\n")

        f.write("[Difficulty]\n")
        f.write(f"HPDrainRate:{hp}\n")
        f.write(f"CircleSize:{cs}\n")
        f.write(f"OverallDifficulty:{od}\n")
        f.write(f"ApproachRate:{ar}\n")
        f.write("SliderMultiplier:1.4\n")
        f.write("SliderTickRate:1\n\n")

        f.write("[TimingPoints]\n")
        f.write(f"0,{ms_per_beat:.6f},4,2,0,100,1,0\n\n")

        f.write("[HitObjects]\n")
        for note in filtrerte:
            note_type = note["type"]
            # Legg til ny combo på første note og hver 8. note
            idx = filtrerte.index(note)
            if idx == 0 or idx % 8 == 0:
                note_type += 4  # Ny combo bit

            f.write(f"{note['x']},{note['y']},{note['timestamp']},{note_type},0\n")

    print(f"\nFerdig! Lagret til: {output_path}")
    print(f"Legg {audio_filename} og {Path(output_path).name} i samme mappe i osu! Songs")
    return filtrerte


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generer osu! beatmap fra MP3")
    parser.add_argument("--audio",      required=True,              help="Sti til MP3-fil")
    parser.add_argument("--output",     default="generated.osu",    help="Output .osu fil")
    parser.add_argument("--model",      default="checkpoints/beatmap_model.pth", help="Sti til modell")
    parser.add_argument("--ar",         type=float, default=9.0,    help="Approach Rate (0-10)")
    parser.add_argument("--od",         type=float, default=8.0,    help="Overall Difficulty (0-10)")
    parser.add_argument("--cs",         type=float, default=4.0,    help="Circle Size (0-7)")
    parser.add_argument("--hp",         type=float, default=5.0,    help="HP Drain Rate (0-10)")
    parser.add_argument("--threshold",  type=float, default=0.3,    help="Note threshold (0-1)")
    parser.add_argument("--title",      default="AI Generated",     help="Sangtittel")
    parser.add_argument("--artist",     default="Unknown",          help="Artist")
    parser.add_argument("--version",    default="AI Hard",          help="Difficulty navn")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Last inn modell
    model = BeatmapGenerator().to(device)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Modell lastet: {args.model}")
    else:
        print(f"ADVARSEL: Ingen modell funnet på {args.model} – bruker utrente vekter")

    generer_map(
        model      = model,
        audio_path = args.audio,
        output_path = args.output,
        device     = device,
        ar         = args.ar,
        od         = args.od,
        cs         = args.cs,
        hp         = args.hp,
        threshold  = args.threshold,
        title      = args.title,
        artist     = args.artist,
        version    = args.version,
    )