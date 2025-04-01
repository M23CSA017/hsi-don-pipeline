# # import numpy as np
# # import pandas as pd
# # import pytest
# # from src.evaluation.evaluate import evaluate_model, plot_results, shap_analysis
# # from xgboost import XGBRegressor

# # @pytest.fixture
# # def sample_data():
# #     X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
# #     y_train = pd.Series(np.random.rand(100))
# #     X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"feature_{i}" for i in range(5)])
# #     y_test = pd.Series(np.random.rand(20))
# #     return X_train, y_train, X_test, y_test

# # def test_evaluate_model(sample_data):
# #     X_train, y_train, X_test, y_test = sample_data
# #     model = XGBRegressor(n_estimators=10, max_depth=3)
# #     metrics, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    
# #     assert "Test_RMSE" in metrics
# #     assert metrics["Test_RMSE"] > 0
# #     assert len(y_pred) == len(y_test)

# # def test_plot_results(sample_data):
# #     _, _, X_test, y_test = sample_data
# #     y_pred = np.random.rand(20)
# #     # Save plot to file instead of displaying
# #     plot_results(y_test, y_pred, model_name="Test_Model", save_path="data/plots/test_plot.png")

# # def test_shap_analysis(sample_data):
# #     X_train, y_train, _, _ = sample_data
# #     model = XGBRegressor(n_estimators=10, max_depth=3)
# #     model.fit(X_train, y_train)
# #     shap_analysis(model, X_train, plot_save_path="data/plots/")

# import numpy as np
# import pandas as pd
# import pytest
# from src.evaluation.evaluate import evaluate_model, plot_results, shap_analysis
# from xgboost import XGBRegressor

# @pytest.fixture
# def sample_data():
#     X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
#     y_train = pd.Series(np.random.rand(100))
#     X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"feature_{i}" for i in range(5)])
#     y_test = pd.Series(np.random.rand(20))
#     return X_train, y_train, X_test, y_test

# def test_evaluate_model(sample_data):
#     X_train, y_train, X_test, y_test = sample_data
#     model = XGBRegressor(n_estimators=10, max_depth=3)
#     metrics, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    
#     assert "Test_RMSE" in metrics
#     assert metrics["Test_RMSE"] > 0
#     assert len(y_pred) == len(y_test)

# def test_plot_results(sample_data):
#     # Generate a dummy prediction vector
#     _, _, X_test, y_test = sample_data
#     y_pred = np.random.rand(len(y_test))
#     # Save plot to file (this test checks that the plot function executes without error)
#     plot_results(y_test, y_pred, model_name="Test_Model", save_path="data/plots/test_plot.png")

# def test_shap_analysis(sample_data):
#     X_train, y_train, _, _ = sample_data
#     model = XGBRegressor(n_estimators=10, max_depth=3)
#     model.fit(X_train, y_train)
#     # Run SHAP analysis; this test ensures no errors occur during plotting.
#     shap_analysis(model, X_train, plot_save_path="data/plots/")

import numpy as np
import pandas as pd
import pytest
from src.evaluation.evaluate import evaluate_model, plot_results, shap_analysis
from xgboost import XGBRegressor

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y_train = pd.Series(np.random.rand(100))
    X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"feature_{i}" for i in range(5)])
    y_test = pd.Series(np.random.rand(20))
    return X_train, y_train, X_test, y_test

def test_evaluate_model(sample_data):
    X_train, y_train, X_test, y_test = sample_data
    model = XGBRegressor(n_estimators=10, max_depth=3)
    metrics, y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    assert "Test_RMSE" in metrics
    assert metrics["Test_RMSE"] > 0
    assert len(y_pred) == len(y_test)

def test_plot_results(sample_data):
    _, _, X_test, y_test = sample_data
    y_pred = np.random.rand(len(y_test))
    # Save plot to file (this test ensures the plot function executes without error)
    plot_results(y_test, y_pred, model_name="Test_Model", save_path="data/plots/test_plot.png")

def test_shap_analysis(sample_data):
    X_train, y_train, _, _ = sample_data
    model = XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(X_train, y_train)
    # This test ensures SHAP analysis runs without errors.
    shap_analysis(model, X_train, plot_save_path="data/plots/")
