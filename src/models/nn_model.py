import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import optuna
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------
#  DONRegressor Model
# --------------------------------------------------------


class DONRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        """
        PyTorch Neural Network for DON Regression with dynamic layers and dropout.
        :param input_dim: Number of input features.
        :param hidden_dim: Number of hidden neurons (default: 128).
        :param num_layers: Number of hidden layers (default: 3).
        :param dropout: Dropout rate for regularization.
        """
        super(DONRegressor, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------
#  Model Training & Evaluation
# --------------------------------------------------------
def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_path,
        num_epochs=100,
        patience=10):
    """
    Train the PyTorch model with early stopping.

    :param model: DONRegressor instance.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param criterion: Loss function (e.g., MSE).
    :param optimizer: Optimizer (e.g., Adam, SGD).
    :param model_path: Path to save the best model.
    :param num_epochs: Maximum number of epochs.
    :param patience: Early stopping patience threshold.
    """
    best_loss = float('inf')
    no_improve_epochs = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Move batch to correct device
            X_batch, y_batch = [x.to(device) for x in batch]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader)
        val_loss = val_metrics['Test_RMSE']

        # Early stopping with tolerance (delta)
        if best_loss - val_loss > 1e-4:  # Small delta for numerical precision
            best_loss = val_loss
            no_improve_epochs = 0
            save_model(model, model_path)
            logger.info(
                f"Improved Validation RMSE: {val_loss:.4f}, Model Saved!")
        else:
            no_improve_epochs += 1

        # Reduce LR on plateau
        scheduler.step(val_loss)

        # Save intermediate model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"./data/models/intermediate_model_epoch_{epoch + 1}.pt"
            save_model(model, checkpoint_path)

        # Early stopping if no improvement for 'patience' epochs
        if no_improve_epochs >= patience:
            logger.info("Early stopping triggered due to no improvement.")
            break


def evaluate_model(model, data_loader):
    """
    Evaluate PyTorch model performance.
    """
    model.eval()
    y_true, y_pred = [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in data_loader:
            X_batch, y_batch = [x.to(device) for x in batch]
            y_pred_batch = model(X_batch)
            y_true.extend(y_batch.cpu().numpy().flatten())
            y_pred.extend(y_pred_batch.cpu().numpy().flatten())

    # Calculate metrics
    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    r2 = 1 - (np.sum((np.array(y_true) - np.array(y_pred)) ** 2) /
              np.sum((np.array(y_true) - np.mean(y_true)) ** 2))

    return {
        "Test_RMSE": rmse,
        "Test_MAE": mae,
        "Test_R2": r2,
        "Test_Predictions": np.array(y_pred)}


def save_model(model, path):
    """
    Save the trained PyTorch model.

    :param model: Trained PyTorch model.
    :param path: Path to save the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f" Model saved to {path}")


def load_model(model, path):
    """
    Load PyTorch model from specified path.

    :param model: PyTorch model instance.
    :param path: Path to model file.
    """
    if os.path.exists(path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)  # Ensure model is sent to the correct device
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model
    else:
        raise FileNotFoundError(f"❗ Model file not found: {path}")


# --------------------------------------------------------
# Optuna Integration for Hyperparameter Tuning
# --------------------------------------------------------
def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function to optimize PyTorch neural network.

    :param trial: Optuna trial object.
    :param X_train: Training data.
    :param y_train: Training target.
    :param X_val: Validation data.
    :param y_val: Validation target.
    :return: Validation RMSE.
    """
    # Hyperparameter search space
    num_layers = trial.suggest_int('num_layers', 3, 6)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    # Define the model dynamically based on hyperparameters
    model = DONRegressor(
        X_train.shape[1],
        hidden_dim,
        num_layers,
        dropout_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    # Prepare DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train),
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # Dynamic path for trial-specific model
    trial_model_path = f"./data/models/nn_model_trial_{trial.number}.pt"

    # Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_path=trial_model_path,
        num_epochs=100,
        patience=10)

    # Evaluate on validation data
    val_metrics = evaluate_model(model, val_loader)
    val_rmse = val_metrics['Test_RMSE']

    return val_rmse


def tune_nn_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Run Optuna hyperparameter tuning for PyTorch Neural Network.

    :param X_train: Training data.
    :param y_train: Training target.
    :param X_val: Validation data.
    :param y_val: Validation target.
    :param n_trials: Number of Optuna trials.
    """
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=10))
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_val,
            y_val),
        n_trials=n_trials)

    logger.info(f"🏆 Best trial RMSE: {study.best_value:.4f}")
    logger.info(f"🔧 Best hyperparameters: {study.best_trial.params}")

    # Load the best model dynamically
    best_trial_model_path = f"./data/models/nn_model_trial_{study.best_trial.number}.pt"
    best_model = DONRegressor(
        X_train.shape[1],
        hidden_dim=study.best_trial.params['hidden_dim'],
        num_layers=study.best_trial.params['num_layers'],
        dropout=study.best_trial.params['dropout']
    )

    # Load the best model
    best_model = load_model(best_model, best_trial_model_path)
    return best_model, study.best_trial.params


# --------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------
def run_training_pipeline(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Run the full training pipeline with Optuna tuning.

    :param X_train: Training data.
    :param y_train: Training target.
    :param X_val: Validation data.
    :param y_val: Validation target.
    :param n_trials: Number of Optuna trials.
    """
    best_model, best_params = tune_nn_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=n_trials)

    # Save the final best NN model
    save_model(best_model, "./data/models/best_nn_model.pt")

    # Evaluate and report final model performance
    best_batch_size = best_params['batch_size']
    val_loader = DataLoader(
        TensorDataset(
            X_val,
            y_val),
        batch_size=best_batch_size)
    final_metrics = evaluate_model(best_model, val_loader)
    logger.info("Final Model Performance:")
    logger.info(final_metrics)
