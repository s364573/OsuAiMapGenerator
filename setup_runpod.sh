#!/bin/bash
# setup_runpod.sh - Kjør én gang etter pod-oppstart
# Forutsetter at Network Volume er mountet på /workspace

set -e
echo "=== osu! AI Beatmap Generator – RunPod Setup ==="

cd /workspace

# ── Clone eller oppdater repo ──
if [ -d "OsuAiMapGenerator" ]; then
    echo "Repo finnes – puller siste versjon..."
    cd OsuAiMapGenerator
    git pull origin master
    cd /workspace
else
    echo "Kloner repo..."
    git clone https://github.com/s364573/OsuAiMapGenerator.git
fi

cd /workspace/OsuAiMapGenerator

# ── Installer avhengigheter ──
echo "Installerer pakker..."
pip install -q torch torchaudio librosa mlflow matplotlib \
    torchmetrics scikit-image scipy requests --ignore-installed blinker

# ── Sjekk CUDA ──
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Lag mapper ──
mkdir -p /workspace/OsuAiMapGenerator/checkpoints
mkdir -p /workspace/Songs

# ── Sjekk Songs ──
SONG_COUNT=$(find /workspace/Songs -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "Songs-mapper funnet: $SONG_COUNT"

# ── Generer JSON hvis Songs finnes ──
if [ "$SONG_COUNT" -gt "0" ]; then
    echo "Genererer song_data.json..."
    python3 src/dataloading/load_data.py

    echo "Filtrerer hard maps..."
    python3 -c "
import json
with open('song_data.json') as f:
    data = json.load(f)
hard = []
for song in data:
    bms = [b for b in song['beatmaps']
           if (b['difficulty']['ar'] + b['difficulty']['od']) / 2 > 7]
    if bms:
        hard.append({**song, 'beatmaps': bms})
with open('hard_data.json', 'w') as f:
    json.dump(hard, f, indent=2)
print(f'Hard maps: {sum(len(s[\"beatmaps\"]) for s in hard)} fra {len(hard)} sanger')
"
else
    echo "Ingen Songs funnet. Last opp sanger til /workspace/Songs/ først."
fi

echo ""
echo "=== Setup ferdig! ==="
echo ""
echo "Neste steg:"
echo "  1. Last opp sanger: scp -r Songs/ root@IP:/workspace/Songs/"
echo "  2. Start trening:   python src/train.py --data_path hard_data.json"
echo "  3. MLflow UI:       mlflow ui --backend-store-uri sqlite:///mlflow.db"