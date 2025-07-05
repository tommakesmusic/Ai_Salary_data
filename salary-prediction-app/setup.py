from setuptools import setup, find_packages

setup(
    name="ai-salary-prediction-app",
    version="1.0.0",
    description="AI Salary Prediction and Insights Streamlit Application",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0", 
        "numpy>=1.21.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0"
    ],
    python_requires=">=3.8",
)