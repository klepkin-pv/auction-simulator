"""
CTR-модель на основе логистической регрессии и нейросети (PyTorch).

Предсказываем вероятность клика по фичам: позиция, категория,
качество пользователя, ставка рекламодателя.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ── Подготовка фичей ──────────────────────────────────────────────────────────

def build_features(impressions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = impressions.copy()

    le = LabelEncoder()
    df["category_enc"] = le.fit_transform(df["category"])

    feature_cols = ["rank", "bid", "user_quality", "category_enc"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["click"].values.astype(np.float32)
    return X, y


# ── Логистическая регрессия (baseline) ───────────────────────────────────────

class LogRegCTR:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=500)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict_proba(X)
        return {
            "auc": roc_auc_score(y, preds),
            "log_loss": log_loss(y, preds),
        }


# ── Нейросеть (PyTorch) ───────────────────────────────────────────────────────

class CTRNet(nn.Module):
    def __init__(self, input_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralCTR:
    def __init__(self, epochs: int = 20, lr: float = 1e-3, batch_size: int = 512):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = CTRNet()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_s = self.scaler.fit_transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_s)
        y_t = torch.from_numpy(y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_t))
            epoch_loss = 0.0
            for i in range(0, len(X_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model(X_t[idx]), y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}  loss={epoch_loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_s = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            return self.model(torch.from_numpy(X_s)).numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict_proba(X)
        return {
            "auc": roc_auc_score(y, preds),
            "log_loss": log_loss(y, preds),
        }


# ── Точка входа ───────────────────────────────────────────────────────────────

def train_and_compare(impressions: pd.DataFrame):
    X, y = build_features(impressions)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=== Logistic Regression (baseline) ===")
    lr_model = LogRegCTR()
    lr_model.fit(X_train, y_train)
    print(lr_model.evaluate(X_test, y_test))

    print("\n=== Neural CTR Model ===")
    nn_model = NeuralCTR(epochs=20)
    nn_model.fit(X_train, y_train)
    print(nn_model.evaluate(X_test, y_test))

    return lr_model, nn_model
