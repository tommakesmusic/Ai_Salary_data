import joblib
import json
import os
import pandas as pd

def load_salary_model(model_path):
    """
    Load the trained XGBoost salary prediction model
    
    Args:
        model_path (str): Path to the saved model file
    
    Returns:
        model: Loaded XGBoost model
        metadata: Model metadata dictionary
    """
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.joblib', '_metadata.json').replace('xgboost_salary_model_', 'model_metadata_')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return model, metadata
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def load_feature_info(feature_info_path):
    """
    Load feature information for the model
    
    Args:
        feature_info_path (str): Path to the feature info JSON file
    
    Returns:
        dict: Feature information dictionary
    """
    try:
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        return feature_info
    except Exception as e:
        print(f"Error loading feature info: {e}")
        return {}

def load_dataset(data_path):
    """
    Load the dataset
    
    Args:
        data_path (str): Path to the dataset CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None