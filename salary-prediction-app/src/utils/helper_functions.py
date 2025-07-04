import numpy as np
import pandas as pd

def predict_salary(model, feature_values, feature_names):
    """
    Predict salary using the trained model
    
    Args:
        model: Trained XGBoost model
        feature_values: Dictionary of feature values
        feature_names: List of expected feature names
    
    Returns:
        predicted_salary: Predicted salary in USD
        log_prediction: Log-transformed prediction
    """
    try:
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # Make prediction (returns log-transformed value)
        log_prediction = model.predict(input_df)[0]
        
        # Convert back to actual salary scale
        predicted_salary = np.expm1(log_prediction)
        
        return predicted_salary, log_prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

def get_business_insights(df, query_type, **kwargs):
    """
    Generate business insights based on query type
    
    Args:
        df: DataFrame with salary data
        query_type: Type of insight to generate
        **kwargs: Additional parameters for specific queries
    
    Returns:
        dict: Insights data
    """
    insights = {}
    
    if query_type == "top_skills_by_hub":
        hub = kwargs.get('hub')
        insights = get_top_skills_by_hub(df, hub)
    elif query_type == "salary_trends":
        insights = get_salary_trends(df)
    elif query_type == "demand_analysis":
        insights = get_demand_analysis(df)
    
    return insights

def get_top_skills_by_hub(df, hub):
    """Get top paying skills for a specific hub"""
    hub_col = f'Hub_{hub}'
    if hub_col not in df.columns:
        return {}
    
    hub_data = df[df[hub_col] == 1]
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    
    skills_data = []
    for skill_col in skill_columns:
        skill_name = skill_col.replace('skill_', '')
        skill_jobs = hub_data[hub_data[skill_col] == 1]
        if len(skill_jobs) > 0:
            avg_salary = skill_jobs['salary_usd'].mean() if 'salary_usd' in df.columns else 0
            count = len(skill_jobs)
            skills_data.append({
                'skill': skill_name,
                'avg_salary': avg_salary,
                'count': count
            })
    
    return {'skills': sorted(skills_data, key=lambda x: x['avg_salary'], reverse=True)}

def get_salary_trends(df):
    """Get salary trends analysis"""
    trends = {}
    
    if 'experience_level' in df.columns and 'salary_usd' in df.columns:
        exp_salaries = df.groupby('experience_level')['salary_usd'].agg(['mean', 'median', 'count']).to_dict()
        trends['by_experience'] = exp_salaries
    
    if 'company_size' in df.columns and 'salary_usd' in df.columns:
        size_salaries = df.groupby('company_size')['salary_usd'].agg(['mean', 'median', 'count']).to_dict()
        trends['by_company_size'] = size_salaries
    
    return trends

def get_demand_analysis(df):
    """Analyze skill demand across the market"""
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    
    demand_data = []
    for skill_col in skill_columns:
        skill_name = skill_col.replace('skill_', '')
        demand = df[skill_col].sum()
        percentage = (demand / len(df)) * 100
        
        demand_data.append({
            'skill': skill_name,
            'demand': demand,
            'percentage': percentage
        })
    
    return {'skills_demand': sorted(demand_data, key=lambda x: x['demand'], reverse=True)}

def format_salary(salary):
    """Format salary for display"""
    if salary >= 1000000:
        return f"${salary/1000000:.1f}M"
    elif salary >= 1000:
        return f"${salary/1000:.0f}K"
    else:
        return f"${salary:.0f}"

def calculate_percentile(values, target_value):
    """Calculate what percentile a target value falls into"""
    return (np.array(values) <= target_value).mean() * 100