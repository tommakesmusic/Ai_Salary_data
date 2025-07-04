import joblib
import json
import os
import pandas as pd
import numpy as np

def load_model_from_directory(models_dir):
    """
    Load the most recent model from the models directory
    
    Args:
        models_dir (str): Path to the models directory
    
    Returns:
        tuple: (model, metadata) or (None, None) if not found
    """
    try:
        if not os.path.exists(models_dir):
            return None, None
        
        # Find all model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        
        if not model_files:
            return None, None
        
        # Get the most recent model
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)
        
        # Load the model
        model = joblib.load(model_path)
        
        # Load metadata
        metadata_file = latest_model.replace('.joblib', '.json').replace('xgboost_salary_model_', 'model_metadata_')
        metadata_path = os.path.join(models_dir, metadata_file)
        
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_model_info(models_dir):
    """
    Get information about available models
    
    Args:
        models_dir (str): Path to the models directory
    
    Returns:
        dict: Model information
    """
    info = {
        'models_found': 0,
        'model_files': [],
        'metadata_files': [],
        'feature_files': []
    }
    
    if not os.path.exists(models_dir):
        return info
    
    files = os.listdir(models_dir)
    
    info['model_files'] = [f for f in files if f.endswith('.joblib')]
    info['metadata_files'] = [f for f in files if f.startswith('model_metadata_')]
    info['feature_files'] = [f for f in files if f.startswith('feature_info_')]
    info['models_found'] = len(info['model_files'])
    
    return info

def validate_model_files(models_dir):
    """
    Validate that all necessary model files are present
    
    Args:
        models_dir (str): Path to the models directory
    
    Returns:
        dict: Validation results
    """
    validation = {
        'valid': False,
        'model_exists': False,
        'metadata_exists': False,
        'feature_info_exists': False,
        'messages': []
    }
    
    if not os.path.exists(models_dir):
        validation['messages'].append(f"Models directory not found: {models_dir}")
        return validation
    
    files = os.listdir(models_dir)
    
    # Check for model files
    model_files = [f for f in files if f.endswith('.joblib')]
    if model_files:
        validation['model_exists'] = True
        validation['messages'].append(f"✅ Found {len(model_files)} model file(s)")
    else:
        validation['messages'].append("❌ No model files (.joblib) found")
    
    # Check for metadata files
    metadata_files = [f for f in files if f.startswith('model_metadata_')]
    if metadata_files:
        validation['metadata_exists'] = True
        validation['messages'].append(f"✅ Found {len(metadata_files)} metadata file(s)")
    else:
        validation['messages'].append("⚠️ No metadata files found")
    
    # Check for feature info files
    feature_files = [f for f in files if f.startswith('feature_info_')]
    if feature_files:
        validation['feature_info_exists'] = True
        validation['messages'].append(f"✅ Found {len(feature_files)} feature info file(s)")
    else:
        validation['messages'].append("⚠️ No feature info files found")
    
    validation['valid'] = validation['model_exists']
    
    return validation