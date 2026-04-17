#!/bin/bash
# setup_runpod.sh - Kjør én gang etter pod-oppstart
# Setter opp miljø og repo – IKKE data-generering
# Data-generering kjøres manuelt etter at Songs er kopiert over

set -e
echo "=== osu! AI Beatmap Generator – RunPod Setup ==="

cd /workspace

# ── Clone eller oppdater repo ──
if [ -d "OsuAiMapGenerator" ]; then
    echo "Repo finnes – puller siste versjon..."
    cd OsuAiMapGenerator
    git stash
    git pull origin master
    cd /workspace
else
    echo "Kloner repo..."
    git clone https://github.com/s364573/OsuAiMapGenerator.git
fi

cd /workspace/OsuAiMapGenerator

# ── Installer avhengigheter ──
echo "Installerer pakker..."
pip install -q librosa mlflow matplotlib \
    torchmetrics scikit-image scipy requests \
    --ignore-installed blinker \
    --root-user-action ignore

# ── Sjekk CUDA ──
python3 -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Lag mapper ──
mkdir -p /workspace/OsuAiMapGenerator/checkpoints
mkdir -p /workspace/Songs

SONG_COUNT=$(find /workspace/Songs -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "Songs-mapper funnet: $SONG_COUNT"

echo ""
echo "=== Setup ferdig! ==="
echo ""
echo "Neste steg:"
echo ""
echo "  1. Kopier Songs fra din PC:"
echo "     scp -i ~/.ssh/id_ed25519 -P PORT -r Songs/ root@IP:/workspace/Songs/"
echo ""
echo "  2. Når Songs er kopiert, generer data:"
echo "     cd /workspace/OsuAiMapGenerator"
echo "     SONGS_FOLDER=/workspace/Songs python3 src/dataloading/load_data.py"
echo "     python3 src/dataloading/make_hard_data.py"
echo ""
echo "  3. Start trening:"
echo "     python3 src/train.py --data_path hard_data.json --epochs 50"
echo ""
echo "  4. MLflow UI (i egen terminal):"
echo "     mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0"