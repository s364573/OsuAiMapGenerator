"""
model.py - BeatmapGenerator v2
Input:  (B, 3, 128, T)  mel + onset + tempogram
Output: (B, T, 7)       [x, y, circle, slider_prob, slider_end_x, slider_end_y, spinner]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# OUTPUT INDEKSER (sync med dataset.py)
# ─────────────────────────────────────────────
IDX_X            = 0
IDX_Y            = 1
IDX_CIRCLE       = 2
IDX_SLIDER_PROB  = 3
IDX_SLIDER_END_X = 4
IDX_SLIDER_END_Y = 5
IDX_SPINNER      = 6
N_OUTPUTS        = 7


# ─────────────────────────────────────────────
# CNN ENCODER
# 4 conv-blokker, 3-kanal input
# ─────────────────────────────────────────────
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, dropout=0.3):
        super().__init__()

        self.block1 = self._block(in_channels, 32,  dropout)
        self.block2 = self._block(32,          64,  dropout)
        self.block3 = self._block(64,          128, dropout)
        self.block4 = self._block(128,         256, dropout)

    def _block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=(7, 15), padding=(3, 7)),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),   # halver frekvens, behold tid
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        # x: (B, 3, 128, T)
        x = self.block1(x)   # (B,  32, 64, T)
        x = self.block2(x)   # (B,  64, 32, T)
        x = self.block3(x)   # (B, 128, 16, T)
        x = self.block4(x)   # (B, 256,  8, T)
        return x


# ─────────────────────────────────────────────
# KONDISJONERING
# Blander difficulty + BPM inn i CNN-features
# ─────────────────────────────────────────────
class ConditioningLayer(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, features, cond):
        """
        features: (B, T, feature_dim)
        cond:     (B, cond_dim)
        """
        c = self.net(cond).unsqueeze(1)   # (B, 1, feature_dim)
        return features + c


# ─────────────────────────────────────────────
# BILSTM MED RESIDUAL
# ─────────────────────────────────────────────
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        # Projeksjons-lag for residual connection
        self.proj = nn.Linear(input_size, hidden_size * 2) \
                    if input_size != hidden_size * 2 else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out, _   = self.lstm(x)
        return self.dropout(out) + residual


# ─────────────────────────────────────────────
# OUTPUT HEAD
# Separat hode for posisjon og type
# ─────────────────────────────────────────────
class OutputHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Posisjon: x, y, slider_end_x, slider_end_y
        self.pos_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),   # x, y, end_x, end_y
        )

        # Type: circle, slider_prob, spinner
        self.type_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),   # circle, slider, spinner
        )

    def forward(self, x):
        pos  = self.pos_head(x)    # (B, T, 4)
        typ  = self.type_head(x)   # (B, T, 3)

        # Kombiner til full output: [x, y, circle, slider_prob, end_x, end_y, spinner]
        return torch.cat([
            pos[:, :, :2],   # x, y
            typ[:, :, :2],   # circle, slider_prob
            pos[:, :, 2:],   # end_x, end_y
            typ[:, :, 2:],   # spinner
        ], dim=-1)            # (B, T, 7)


# ─────────────────────────────────────────────
# FULL MODELL
# ─────────────────────────────────────────────
class BeatmapGenerator(nn.Module):
    def __init__(
        self,
        in_channels  = 3,      # mel + onset + tempogram
        cond_dim     = 5,      # AR, OD, CS, HP, BPM
        hidden_size  = 768,
        num_layers   = 4,
        dropout      = 0.3,
    ):
        super().__init__()

        # CNN: 256 kanaler × 8 frekvens-buckets = 2048
        self.cnn_features = 256 * 8

        self.encoder      = CNNEncoder(in_channels, dropout)
        self.conditioning = ConditioningLayer(cond_dim, self.cnn_features)
        self.bilstm       = BiLSTMEncoder(
            input_size  = self.cnn_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
        )
        self.output_head  = OutputHead(hidden_size * 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, diff, bpm):
        """
        x:    (B, 3, 128, T)
        diff: (B, 4)   AR, OD, CS, HP normalisert
        bpm:  (B, 1)   BPM normalisert
        """
        # CNN
        x = self.encoder(x)                    # (B, 256, 8, T)

        # Reshape til sekvens
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T, 256, 8)
        x = x.reshape(B, T, C * F)             # (B, T, 2048)

        # Kondisjonering
        cond = torch.cat([diff, bpm], dim=1)   # (B, 5)
        x    = self.conditioning(x, cond)      # (B, T, 2048)

        # BiLSTM
        x = self.bilstm(x)                     # (B, T, hidden*2)

        # Output
        x = self.output_head(x)                # (B, T, 7)

        return x


# ─────────────────────────────────────────────
# LOSS FUNKSJON
# ─────────────────────────────────────────────
class BeatmapLoss(nn.Module):
    def __init__(
        self,
        pos_weight_circle  = 30.0,
        pos_weight_slider  = 40.0,
        pos_weight_spinner = 60.0,
        diversity_weight   = 0.05,
        slider_end_weight  = 0.5,
        device             = "cpu",
    ):
        super().__init__()
        self.diversity_weight  = diversity_weight
        self.slider_end_weight = slider_end_weight

        self.mse = nn.MSELoss()

        self.bce_circle = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_circle]).to(device))
        self.bce_slider = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_slider]).to(device))
        self.bce_spinner = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_spinner]).to(device))

    def forward(self, pred, target):
        """
        pred:   (B, T, 7)
        target: (B, 7, T) → transpose til (B, T, 7)
        """
        target = target.permute(0, 2, 1)  # (B, T, 7)

        # ── Posisjon (x, y) – kun ved noter ──
        note_mask = (
            target[:, :, IDX_CIRCLE] +
            target[:, :, IDX_SLIDER_PROB] +
            target[:, :, IDX_SPINNER]
        ).clamp(0, 1).unsqueeze(-1)    # (B, T, 1)

        pred_xy   = pred[:, :, IDX_X:IDX_Y+1]
        target_xy = target[:, :, IDX_X:IDX_Y+1]
        loss_pos  = self.mse(pred_xy * note_mask, target_xy * note_mask)

        # ── Note-typer ──
        loss_circle  = self.bce_circle(
            pred[:, :, IDX_CIRCLE],
            target[:, :, IDX_CIRCLE])

        loss_slider  = self.bce_slider(
            pred[:, :, IDX_SLIDER_PROB],
            target[:, :, IDX_SLIDER_PROB])

        loss_spinner = self.bce_spinner(
            pred[:, :, IDX_SPINNER],
            target[:, :, IDX_SPINNER])

        # ── Slider end-posisjon – kun der target er slider ──
        slider_mask = target[:, :, IDX_SLIDER_PROB].unsqueeze(-1)
        pred_end    = pred[:, :, IDX_SLIDER_END_X:IDX_SLIDER_END_Y+1]
        target_end  = target[:, :, IDX_SLIDER_END_X:IDX_SLIDER_END_Y+1]
        loss_end    = self.mse(
            pred_end   * slider_mask,
            target_end * slider_mask)

        # ── Diversity loss – straffer hvis noter klumper seg ──
        # Kun regnet over frames med predikert note
        pred_circle_sig = torch.sigmoid(pred[:, :, IDX_CIRCLE])
        note_pred_mask  = (pred_circle_sig > 0.3).float()

        std_x = torch.std(pred[:, :, IDX_X] * note_pred_mask + 0.5 * (1 - note_pred_mask))
        std_y = torch.std(pred[:, :, IDX_Y] * note_pred_mask + 0.5 * (1 - note_pred_mask))
        loss_diversity  = -(std_x + std_y)

        # ── Total loss ──
        loss = (
            loss_pos
            + 0.4 * loss_circle
            + 0.4 * loss_slider
            + 0.2 * loss_spinner
            + self.slider_end_weight * loss_end
            + self.diversity_weight  * loss_diversity
        )

        return loss, {
            "pos":       loss_pos.item(),
            "circle":    loss_circle.item(),
            "slider":    loss_slider.item(),
            "spinner":   loss_spinner.item(),
            "slider_end": loss_end.item(),
            "diversity": loss_diversity.item(),
        }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BeatmapGenerator().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametere: {total:,}")

    B, T = 2, 3000
    x    = torch.randn(B, 3, 128, T).to(device)
    diff = torch.rand(B, 4).to(device)
    bpm  = torch.rand(B, 1).to(device)

    out  = model(x, diff, bpm)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")    # (2, 3000, 7)

    loss_fn = BeatmapLoss(device=device)
    target  = torch.zeros(B, N_OUTPUTS, T).to(device)
    loss, breakdown = loss_fn(out, target)

    print(f"\nLoss: {loss.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\nModell klar!")