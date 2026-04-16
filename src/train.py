"""
train.py - Treningsloop for osu! Beatmap Generator
Bruker BeatmapDataset og BeatmapGenerator med MLflow tracking
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from datetime import timedelta
from torch.utils.data import DataLoader, random_split

# Legg til src i path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloading.dataset import BeatmapDataset, collate_fn
from model.model import BeatmapGenerator, BeatmapLoss
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("librosa").setLevel(logging.ERROR)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_path":    "hard_data.json",
    "model_path":   "checkpoints/beatmap_model.pth",
    "epochs":       50,
    "batch_size":   4,
    "lr":           0.001,
    "lr_step":      5,
    "lr_gamma":     0.5,
    "hidden_size":  512,
    "num_layers":   3,
    "dropout":      0.3,
    "pos_weight":   40.0,
    "val_split":    0.2,
    "early_stop":   7,
    "segment_len":  2000,
    "num_workers":  0,
    "experiment":   "osu-beatmap-generator-v2",
}


# ─────────────────────────────────────────────
# TRENING
# ─────────────────────────────────────────────
def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Lag checkpoint mappe
    os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)

    # Dataset
    full_dataset = BeatmapDataset(
        json_path      = config["data_path"],
        segment_length = config["segment_len"],
    )

    # Train/val split
    val_size   = int(len(full_dataset) * config["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config["batch_size"],
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = config["num_workers"],
    )

    print(f"Train: {train_size} | Val: {val_size}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Modell
    model = BeatmapGenerator(
        hidden_size = config["hidden_size"],
        num_layers  = config["num_layers"],
        dropout     = config["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametere: {total_params:,}")

    # Optimizer og scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = config["lr_step"],
        gamma     = config["lr_gamma"],
    )
    loss_fn = BeatmapLoss(pos_weight=config["pos_weight"], device=device)

    # MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(config["experiment"])

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses     = []
    val_losses       = []

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(config["epochs"]):
            epoch_start = time.time()

            # ── TRENING ──
            model.train()
            total_loss = 0
            antall     = 0

            for batch_idx, batch in enumerate(train_loader):
                if batch is None:
                    continue

                try:
                    x      = batch["input"].to(device).contiguous()
                    target = batch["target"].to(device).contiguous()
                    diff   = batch["diff"].to(device)
                    bpm    = batch["bpm"].to(device)

                    optimizer.zero_grad()
                    pred = model(x, diff, bpm)

                    # Juster lengde
                    min_t    = min(pred.shape[1], target.shape[2])
                    pred     = pred[:, :min_t, :]
                    target_t = target[:, :, :min_t]

                    loss = loss_fn(pred, target_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    antall     += 1

                    # Progress
                    elapsed      = time.time() - epoch_start
                    batches_left = len(train_loader) - (batch_idx + 1)
                    eta          = timedelta(seconds=int(elapsed / (batch_idx + 1) * batches_left))
                    progress     = (batch_idx + 1) / len(train_loader) * 100

                    print(
                        f"\rEpoch {epoch+1}/{config['epochs']} "
                        f"[{progress:5.1f}%] "
                        f"Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"ETA: {eta}",
                        end=""
                    )

                except RuntimeError as e:
                    print(f"\nHopper over batch – {str(e)[:60]}")
                    continue

            train_avg = total_loss / max(antall, 1)
            train_losses.append(train_avg)

            # ── VALIDERING ──
            model.eval()
            val_total  = 0
            val_antall = 0

            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    try:
                        x      = batch["input"].to(device).contiguous()
                        target = batch["target"].to(device).contiguous()
                        diff   = batch["diff"].to(device)
                        bpm    = batch["bpm"].to(device)

                        pred  = model(x, diff, bpm)
                        min_t = min(pred.shape[1], target.shape[2])
                        loss  = loss_fn(pred[:, :min_t], target[:, :, :min_t])

                        val_total  += loss.item()
                        val_antall += 1
                    except:
                        continue

            val_avg = val_total / max(val_antall, 1)
            val_losses.append(val_avg)

            scheduler.step()
            lr        = scheduler.get_last_lr()[0]
            epoch_tid = timedelta(seconds=int(time.time() - epoch_start))

            # MLflow logging
            mlflow.log_metric("train_loss", train_avg, step=epoch)
            mlflow.log_metric("val_loss",   val_avg,   step=epoch)
            mlflow.log_metric("lr",         lr,        step=epoch)

            print(
                f"\nEpoch {epoch+1}/{config['epochs']} | "
                f"Train: {train_avg:.4f} | "
                f"Val: {val_avg:.4f} | "
                f"LR: {lr:.6f} | "
                f"Tid: {epoch_tid}\n"
            )

            # Checkpoint hver 3. epoch
            if (epoch + 1) % 3 == 0:
                ckpt = config["model_path"].replace(".pth", f"_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt)
                print(f"Checkpoint: {ckpt}")

            # Best modell + early stopping
            if val_avg < best_val_loss:
                best_val_loss    = val_avg
                patience_counter = 0
                torch.save(model.state_dict(), config["model_path"])
                print(f"Ny beste modell! Val loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Ingen forbedring ({patience_counter}/{config['early_stop']})")
                if patience_counter >= config["early_stop"]:
                    print("Early stopping!")
                    break

        mlflow.pytorch.log_model(model, "beatmap_model")

    # Graf
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Validering")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()
    print("Ferdig! Graf lagret til loss_curve.png")

    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   default="hard_data.json")
    parser.add_argument("--model_path",  default="checkpoints/beatmap_model.pth")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int,   default=512)
    parser.add_argument("--num_layers",  type=int,   default=3)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--segment_len", type=int,   default=2000)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in vars(args).items() if v is not None})

    train(config)