from typing import Tuple, Dict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from time import time
import joblib

# Configure environment
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
sns.set_palette("husl")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('don_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# 1. Optimized Neural Network Model
# --------------------------
class DONPredictorNN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            
            nn.Linear(64, 1)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# --------------------------
# 2. Optimized Training Function
# --------------------------
def train_nn(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 128,
    patience: int = 100
) -> Tuple[nn.Module, Dict[str, list], StandardScaler, RobustScaler]:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Enhanced target scaling with log transformation
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val_log.reshape(-1, 1)).flatten()
    
    # Feature scaling
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    
    # Data loaders
    train_data = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.FloatTensor(y_train_scaled).reshape(-1, 1)
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    y_val_t = torch.FloatTensor(y_val_scaled).reshape(-1, 1).to(device)
    
    # Model configuration
    model = DONPredictorNN(X_train.shape[1]).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, min_lr=1e-6)
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_weights = model.state_dict()
            torch.save(best_weights, '../data/models/nn_checkpoint.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    model.load_state_dict(torch.load('../data/models/nn_checkpoint.pth'))
    return model, history, feature_scaler, y_scaler

# --------------------------
# 3. Optimized Evaluation
# --------------------------
def evaluate_nn(
    model: nn.Module, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    feature_scaler: StandardScaler,
    y_scaler: RobustScaler
) -> Dict[str, float]:
    
    device = next(model.parameters()).device
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_log = np.log1p(y_test)
    
    with torch.no_grad():
        test_pred_scaled = model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
        test_pred_log = y_scaler.inverse_transform(test_pred_scaled)
        test_pred = np.expm1(test_pred_log)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
            'R2': r2_score(y_test, test_pred),
            'Relative_Error': np.median(np.abs(y_test - test_pred.flatten()) / (y_test + 1))  # Using median for robustness
        }
    
    # Enhanced visualization
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, test_pred, alpha=0.6)
    plt.plot([0, y_test.max()], [0, y_test.max()], 'r--')
    plt.xlabel('Actual DON (ppb)')
    plt.ylabel('Predicted DON (ppb)')
    plt.title(f'Actual vs Predicted (R2: {metrics["R2"]:.2f})')
    
    plt.subplot(1, 3, 2)
    residuals = y_test - test_pred.flatten()
    plt.scatter(test_pred, residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals (MAE: {metrics["MAE"]:.1f})')
    
    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Prediction Error')
    plt.title(f'Error Distribution (RMSE: {metrics["RMSE"]:.1f})')
    
    plt.tight_layout()
    plt.show()
    
    return metrics

if __name__ == "__main__":
    logger.info("Starting Optimized Neural Network Pipeline")
    
    # Load data
    X_train = pd.read_csv("../data/processed/x_train.csv").drop(columns=['hsi_id'])
    X_test = pd.read_csv("../data/processed/x_test.csv").drop(columns=['hsi_id'])
    y_train = pd.read_csv("../data/processed/y_train.csv")['vomitoxin_ppb'].values
    y_test = pd.read_csv("../data/processed/y_test.csv")['vomitoxin_ppb'].values
    
    logger.info(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"Target stats - Min: {y_train.min()}, Max: {y_train.max()}, Mean: {y_train.mean():.1f}, Std: {y_train.std():.1f}")
    
    # Train model
    model, history, feature_scaler, y_scaler = train_nn(
        X_train.values, y_train, X_test.values, y_test
    )
    
    # Evaluate
    metrics = evaluate_nn(model, X_test.values, y_test, feature_scaler, y_scaler)
    
    logger.info("\nFinal Test Metrics:")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
    
    # Save artifacts
    torch.save(model.state_dict(), '../data/models/nn_model_optimized.pth')
    joblib.dump(feature_scaler, '../data/models/feature_scaler.pkl')
    joblib.dump(y_scaler, '../data/models/target_scaler.pkl')
    logger.info("Pipeline completed successfully!")