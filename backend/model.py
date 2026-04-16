"""
NEON-FORECAST AI
================
File: backend/model.py
Phase 3 — LSTM Model Architecture

This file defines the neural network.
It is imported by train.py and main.py (FastAPI).

Architecture:
  Input  → (batch, seq_length=14, features=1)
  LSTM   → 2 stacked layers, 64 hidden units each
  Dropout → 20% (prevents overfitting)
  Linear → single output (next day prediction)
  Output → (batch, 1)
"""

import torch
import torch.nn as nn


class NeonForecastLSTM(nn.Module):
    """
    2-layer stacked LSTM for time series forecasting.

    Args:
        input_size  : number of features per time step (1 = just sales units)
        hidden_size : number of memory cells per LSTM layer (64)
        num_layers  : how many LSTM layers stacked (2)
        dropout     : dropout rate between layers (0.2)
        output_size : how many days ahead to predict (1)
    """

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.2,
        output_size: int = 1,
    ):
        super(NeonForecastLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── LSTM layers ───────────────────────────────────────────────────────
        # batch_first=True means input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
            batch_first = True,
        )

        # ── Dropout layer (applied after LSTM output) ─────────────────────────
        self.dropout = nn.Dropout(p=dropout)

        # ── Fully connected output layer ──────────────────────────────────────
        # Takes last hidden state → predicts next day value
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x shape: (batch_size, seq_length, input_size)
        returns: (batch_size, 1)
        """
        batch_size = x.size(0)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Take only the last time step output
        # out[:, -1, :] shape: (batch_size, hidden_size)
        out = self.dropout(out[:, -1, :])

        # Final prediction
        # shape: (batch_size, output_size=1)
        out = self.fc(out)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — run this file directly to verify the model works
# python backend/model.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  NeonForecastLSTM — Architecture Check")
    print("=" * 50)

    model = NeonForecastLSTM(
        input_size  = 1,
        hidden_size = 64,
        num_layers  = 2,
        dropout     = 0.2,
        output_size = 1,
    )

    print(model)
    print()

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters : {total_params:,}")

    # Test with a dummy batch
    # Simulates: batch of 32 samples, each with 14 days of 1 feature
    dummy_input = torch.randn(32, 14, 1)
    output      = model(dummy_input)

    print(f"  Input shape  : {list(dummy_input.shape)}")
    print(f"  Output shape : {list(output.shape)}")
    print()
    print("  Model is working correctly!")
    print("=" * 50)