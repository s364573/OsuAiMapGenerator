"""
model.py - BeatmapGenerator modell
CNN + BiLSTM med difficulty og BPM kondisjonering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# CNN ENCODER
# Ekstraherer lokale mønstre fra spektrogrammet
# ─────────────────────────────────────────────
class CNNEncoder(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.block1 = self._conv_block(1,   32, dropout)
        self.block2 = self._conv_block(32,  64, dropout)
        self.block3 = self._conv_block(64, 128, dropout)

    def _conv_block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(7, 15), padding=(3, 7)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Halver frekvens, behold tid
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        x = self.block1(x)  # (B, 32,  64, T)
        x = self.block2(x)  # (B, 64,  32, T)
        x = self.block3(x)  # (B, 128, 16, T)
        return x


# ─────────────────────────────────────────────
# KONDISJONERING
# Blander difficulty og BPM inn i modellen
# ─────────────────────────────────────────────
class ConditioningLayer(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        """
        cond_dim:   størrelse på kondisjonerings-vektor (4 diff + 1 bpm = 5)
        hidden_dim: størrelse på CNN output features
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cnn_features, cond_vector):
        """
        cnn_features: (B, T, features)
        cond_vector:  (B, cond_dim)
        """
        cond = self.fc(cond_vector)          # (B, hidden_dim)
        cond = cond.unsqueeze(1)             # (B, 1, hidden_dim)
        cond = cond.expand_as(cnn_features)  # (B, T, hidden_dim)
        return cnn_features + cond           # Additivt blend


# ─────────────────────────────────────────────
# BILSTM TEMPORAL ENCODER
# Forstår temporal sammenheng begge veier
# ─────────────────────────────────────────────
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.dropout(x)


# ─────────────────────────────────────────────
# OUTPUT HEAD
# Mapper fra LSTM features til note-prediksjoner
# ─────────────────────────────────────────────
class OutputHead(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 5),  # x, y, circle, slider, spinner
        )

    def forward(self, x):
        return self.fc(x)


# ─────────────────────────────────────────────
# FULL MODELL
# ─────────────────────────────────────────────
class BeatmapGenerator(nn.Module):
    def __init__(
        self,
        cond_dim    = 5,     # AR, OD, CS, HP + BPM
        hidden_size = 512,
        num_layers  = 3,
        dropout     = 0.3,
    ):
        super().__init__()

        # CNN features: 128 kanaler × 16 frekvens-buckets = 2048
        self.cnn_out_features = 128 * 16

        self.encoder     = CNNEncoder(dropout)
        self.conditioning = ConditioningLayer(cond_dim, self.cnn_out_features)
        self.bilstm      = BiLSTMEncoder(
            input_size  = self.cnn_out_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
        )
        self.output_head = OutputHead(hidden_size * 2)  # *2 for bidireksjonell

    def forward(self, x, diff, bpm):
        """
        x:    (B, 1, 128, T) spektrogram
        diff: (B, 4)          AR, OD, CS, HP normalisert
        bpm:  (B, 1)          BPM normalisert
        """
        # CNN
        x = self.encoder(x)              # (B, 128, 16, T)

        # Reshape til sekvens
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)       # (B, T, C, F)
        x = x.reshape(B, T, C * F).contiguous()  # (B, T, 2048)

        # Kondisjonering
        cond = torch.cat([diff, bpm], dim=1)  # (B, 5)
        x    = self.conditioning(x, cond)      # (B, T, 2048)

        # BiLSTM
        x = self.bilstm(x)               # (B, T, hidden*2)

        # Output
        x = self.output_head(x)          # (B, T, 5)

        return x


# ─────────────────────────────────────────────
# LOSS FUNKSJON
# ─────────────────────────────────────────────
class BeatmapLoss(nn.Module):
    def __init__(self, pos_weight=40.0, device="cpu"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )

    def forward(self, pred, target):
        """
        pred:   (B, T, 5)
        target: (B, 5, T) → transpose til (B, T, 5)
        """
        target = target.permute(0, 2, 1)  # (B, T, 5)

        loss_pos  = self.mse(pred[:, :, :2], target[:, :, :2])   # x, y
        loss_type = self.bce(pred[:, :, 2:], target[:, :, 2:])   # circle, slider, spinner

        return loss_pos + 0.5 * loss_type


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BeatmapGenerator().to(device)

    # Tell parametere
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametere: {total:,}")

    # Test forward pass
    B, T = 2, 1000
    x    = torch.randn(B, 1, 128, T).to(device)
    diff = torch.rand(B, 4).to(device)
    bpm  = torch.rand(B, 1).to(device)

    output = model(x, diff, bpm)
    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")  # Skal være (2, 1000, 5)

    # Test loss
    loss_fn = BeatmapLoss(device=device)
    target  = torch.zeros(B, 5, T).to(device)
    loss    = loss_fn(output, target)
    print(f"Loss:   {loss.item():.4f}")

    print("\nModell klar!")