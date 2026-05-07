"""LSTM model with dual prediction heads (classification + regression)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)


class _LSTMNet(nn.Module):
    """LSTM network with dual output heads."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        direction = self.direction_head(last_hidden).squeeze(-1)
        regression = self.regression_head(last_hidden).squeeze(-1)
        return direction, regression


class LSTMModel:
    """LSTM with dual heads for direction classification and pct move regression."""

    def __init__(
        self,
        input_size: int = 80,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._device = torch.device("cpu")
        self._net: _LSTMNet | None = None

    @property
    def name(self) -> str:
        return "lstm"

    @property
    def n_features(self) -> int | None:
        if self._net is not None:
            return self._net.lstm.input_size
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
        self._net = _LSTMNet(input_size, self._hidden_size, self._num_layers).to(self._device)

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
                y_dir_batch = y_dir_batch.to(self._device)
                y_pct_batch = y_pct_batch.to(self._device)

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
            torch.save(self._net.state_dict(), str(p / "lstm.pt"))
            config = {
                "input_size": self._net.lstm.input_size,
                "hidden_size": self._hidden_size,
                "num_layers": self._num_layers,
            }
            with open(p / "config.json", "w") as f:
                json.dump(config, f)

    def load(self, path: str) -> None:
        p = Path(path)
        with open(p / "config.json") as f:
            config = json.load(f)
        self._net = _LSTMNet(
            config["input_size"], config["hidden_size"], config["num_layers"]
        ).to(self._device)
        self._net.load_state_dict(torch.load(str(p / "lstm.pt"), map_location=self._device, weights_only=True))
