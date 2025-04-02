import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from torch.utils.data import TensorDataset, DataLoader


PLOT_DIR = "./data/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)


def perform_residual_analysis(y_true, y_pred, model_name):
    """
    Plot residuals (true - predicted) and save the figure.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Predicted)")
    plt.title(f"Residual Plot: {model_name}")
    plt.grid(True)

    path = os.path.join(
        PLOT_DIR,
        f"residuals_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def run_kfold_cross_validation(
        model_class,
        X,
        y,
        k=5,
        model_type="xgb",
        **kwargs):
    """
    Perform k-fold cross-validation for XGBoost or PyTorch NN model.
    Returns average RMSE and R^2.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores, r2_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_type == "xgb":
            model = model_class(**kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        elif model_type == "nn":
            model = model_class(**kwargs)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(
                y_train.reshape(-1, 1), dtype=torch.float32)
            train_loader = DataLoader(
                TensorDataset(
                    X_train_tensor,
                    y_train_tensor),
                batch_size=32,
                shuffle=True)

            model.train()
            for epoch in range(30):  # Fixed small epochs for CV
                for Xb, yb in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(Xb), yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(
                    torch.tensor(
                        X_val,
                        dtype=torch.float32)).cpu().numpy().flatten()
        else:
            raise ValueError("Invalid model_type. Choose 'xgb' or 'nn'.")

        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2_scores.append(r2_score(y_val, y_pred))

    return {
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
    }
