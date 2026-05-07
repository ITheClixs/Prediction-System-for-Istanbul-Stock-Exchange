"""Transformer model with dual prediction heads (classification + regression)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(
        self, input_size: int, d_model: int = 64, nhead: int = 4,
        num_layers: int = 2, dim_feedforward: int = 128,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        summary = x[:, -1, :]
        direction = self.direction_head(summary).squeeze(-1)
        regression = self.regression_head(summary).squeeze(-1)
        return direction, regression


class TransformerModel:
    """Transformer with dual heads for direction classification and pct move regression."""

    def __init__(
        self,
        input_size: int = 80,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        self._input_size = input_size
        self._d_model = d_model
        self._nhead = nhead
        self._num_layers = num_layers
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._device = torch.device("cpu")
        self._net: _TransformerNet | None = None

    @property
    def name(self) -> str:
        return "transformer"

    @property
    def n_features(self) -> int | None:
        if self._net is not None:
            return self._net.input_projection.in_features
        return self._input_size

    def train(
        self,
        X_train: NDArray[np.float64],
        y_dir_train: NDArray[np.int64],
        y_pct_train: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_dir_val: NDArray[np.int64] | None = None,
        y_pct_val: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        input_size = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
        self._input_size = input_size
        self._net = _TransformerNet(
            input_size, self._d_model, self._nhead, self._num_layers,
        ).to(self._device)

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_dir_t = torch.tensor(y_dir_train, dtype=torch.float32)
        y_pct_t = torch.tensor(y_pct_train, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_dir_t, y_pct_t)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        self._net.train()
        for _ in range(self._epochs):
            for X_batch, y_dir_batch, y_pct_batch in loader:
                X_batch = X_batch.to(self._device)
                dir_pred, pct_pred = self._net(X_batch)
                loss = bce_loss(dir_pred, y_dir_batch) + mse_loss(pct_pred, y_pct_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics: dict[str, float] = {}
        if X_val is not None and y_dir_val is not None and y_pct_val is not None:
            probs, pct_pred = self.predict(X_val)
            pred_dir = (probs > 0.5).astype(int)
            metrics["val_accuracy"] = float(np.mean(pred_dir == y_dir_val))
            metrics["val_mae"] = float(np.mean(np.abs(pct_pred - y_pct_val)))

        return metrics

    def predict(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._net is None:
            raise RuntimeError("Model not trained or loaded")
        self._net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            dir_probs, pct_preds = self._net(X_t)
        return dir_probs.cpu().numpy().astype(np.float64), pct_preds.cpu().numpy().astype(np.float64)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self._net is not None:
            torch.save(self._net.state_dict(), str(p / "transformer.pt"))
            config = {
                "input_size": self._input_size, "d_model": self._d_model,
                "nhead": self._nhead, "num_layers": self._num_layers,
            }
            with open(p / "config.json", "w") as f:
                json.dump(config, f)

    def load(self, path: str) -> None:
        p = Path(path)
        with open(p / "config.json") as f:
            config = json.load(f)
        self._net = _TransformerNet(
            config["input_size"], config["d_model"], config["nhead"], config["num_layers"],
        ).to(self._device)
        self._net.load_state_dict(torch.load(str(p / "transformer.pt"), map_location=self._device, weights_only=True))
