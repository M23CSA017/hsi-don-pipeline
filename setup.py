from setuptools import setup, find_packages

setup(
    name="hsi_don_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "optuna",
        "shap",
        "joblib",
        "fastapi",
        "uvicorn",
        "PyYAML"
    ],
    description="Pipeline for predicting DON concentration using hyperspectral data.",
    author="Your Name",
    author_email="your_email@example.com"
)
