"""
optimize.py

Hyperparameter tuning logic with Optuna for XGBoost.
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold


def objective(trial, X_train, y_train, random_state=42):
    """
    Objective function for Optuna hyperparameter optimization.
    Returns RMSE from cross-validation.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }

    model = xgb.XGBRegressor(
        **params,
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=4
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    score = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=8
    ).mean()

    return np.sqrt(-score)


def run_optuna_optimization(X_train,
                            y_train,
                            random_state=42,
                            n_trials=50) -> optuna.study.Study:
    """
    Run the Optuna study to find best hyperparameters.
    """
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            random_state),
        n_trials=n_trials,
        show_progress_bar=True)
    return study
