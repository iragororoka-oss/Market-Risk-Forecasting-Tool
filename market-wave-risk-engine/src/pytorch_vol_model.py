import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class VolatilityNet(nn.Module):
    """
    Simple neural network that predicts future realized volatility
    from a window of recent daily log returns.
    """

    def __init__(self, lookback: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(lookback, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.model(x)


def create_volatility_dataset(
    returns: pd.Series,
    lookback: int = 30,
    forecast_window: int = 10
):
    """
    Creates supervised learning data.

    X = previous lookback returns
    y = future realized volatility over forecast_window days
    """

    returns = returns.dropna().astype(float).values

    X = []
    y = []

    for i in range(lookback, len(returns) - forecast_window):
        past_returns = returns[i - lookback:i]
        future_returns = returns[i:i + forecast_window]

        future_vol = np.std(future_returns)

        X.append(past_returns)
        y.append(future_vol)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return X, y


def train_volatility_model(
    returns: pd.Series,
    lookback: int = 30,
    forecast_window: int = 10,
    epochs: int = 300,
    learning_rate: float = 0.001
):
    """
    Trains a PyTorch model to forecast future volatility.
    """

    X, y = create_volatility_dataset(
        returns,
        lookback=lookback,
        forecast_window=forecast_window
    )

    if len(X) < 50:
        raise ValueError("Not enough historical data to train the volatility model.")

    split_index = int(len(X) * 0.8)

    X_train = torch.tensor(X[:split_index])
    y_train = torch.tensor(y[:split_index])

    X_test = torch.tensor(X[split_index:])
    y_test = torch.tensor(y[split_index:])

    model = VolatilityNet(lookback)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()

        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        test_predictions = model(X_test)
        test_rmse = torch.sqrt(loss_fn(test_predictions, y_test)).item()

        latest_window = torch.tensor(
            X[-1].reshape(1, -1),
            dtype=torch.float32
        )

        predicted_volatility = model(latest_window).item()

    return {
        "model": model,
        "predicted_volatility": predicted_volatility,
        "test_rmse": test_rmse,
        "lookback": lookback,
        "forecast_window": forecast_window
    }