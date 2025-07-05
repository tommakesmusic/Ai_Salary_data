import streamlit as st
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
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        log_prediction = model.predict(input_df)[0]
        
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

def calculate_salary_percentile(df, predicted_salary):
    """Calculate what percentile the predicted salary falls into"""
    try:

        salary_column = None
        if 'salary_usd' in df.columns:
            salary_column = 'salary_usd'
            return (df[salary_column] <= predicted_salary).mean() * 100
        elif 'log_salary_usd' in df.columns:
            actual_salaries = np.expm1(df['log_salary_usd'])
            return (actual_salaries <= predicted_salary).mean() * 100
        elif 'salary_usd_calculated' in df.columns:
            return (df['salary_usd_calculated'] <= predicted_salary).mean() * 100
        else:
            return 50  # Default if no salary data
    except Exception as e:
        print(f"Error in calculate_salary_percentile: {e}")
        return 50

def compare_to_market(df, predicted_salary, category_col, category_value):
    """Compare predicted salary to category average"""
    try:
        if 'salary_usd' in df.columns:
            market_data = df[df[category_col] == category_value]['salary_usd']
            if len(market_data) > 0:
                market_avg = market_data.mean()
                diff = predicted_salary - market_avg
                return f"${diff:+,.0f}"
        elif 'log_salary_usd' in df.columns:
            market_data = df[df[category_col] == category_value]['log_salary_usd']
            if len(market_data) > 0:
                market_avg = np.expm1(market_data.mean())
                diff = predicted_salary - market_avg
                return f"${diff:+,.0f}"
        elif 'salary_usd_calculated' in df.columns:
            market_data = df[df[category_col] == category_value]['salary_usd_calculated']
            if len(market_data) > 0:
                market_avg = market_data.mean()
                diff = predicted_salary - market_avg
                return f"${diff:+,.0f}"
        
        return "N/A"
    except Exception as e:
        st.warning(f"Error in compare_to_market: {e}")
        return "N/A"

def compare_to_hub(df, predicted_salary, selected_hub):
    """Compare predicted salary to hub average"""
    try:
        hub_col = f'Hub_{selected_hub}'
        
        if hub_col not in df.columns:
            return "N/A"
        
        if 'salary_usd' in df.columns:
            hub_data = df[df[hub_col] == 1]['salary_usd']
            if len(hub_data) > 0:
                hub_avg = hub_data.mean()
                diff = predicted_salary - hub_avg
                return f"${diff:+,.0f}"
        elif 'log_salary_usd' in df.columns:
            hub_data = df[df[hub_col] == 1]['log_salary_usd']
            if len(hub_data) > 0:
                hub_avg = np.expm1(hub_data.mean())
                diff = predicted_salary - hub_avg
                return f"${diff:+,.0f}"
        elif 'salary_usd_calculated' in df.columns:
            hub_data = df[df[hub_col] == 1]['salary_usd_calculated']
            if len(hub_data) > 0:
                hub_avg = hub_data.mean()
                diff = predicted_salary - hub_avg
                return f"${diff:+,.0f}"
        
        return "N/A"
    except Exception as e:
        print(f"Error in compare_to_hub: {e}")
        return "N/A"

def get_hub_distribution(df):
    """Get distribution of jobs by hub"""
    hub_data = []
    hub_columns = [col for col in df.columns if col.startswith('Hub_')]
    
    for hub_col in hub_columns:
        hub_name = hub_col.replace('Hub_', '')
        count = df[hub_col].sum()
        if count > 0:
            hub_data.append({'hub': hub_name, 'count': count})
    
    return pd.DataFrame(hub_data)

def compare_to_category(df, predicted_salary, category_col, category_value):
    """Compare predicted salary to category average"""
    try:
        salary_column = None
        if 'salary_usd' in df.columns:
            salary_column = 'salary_usd'
        elif 'log_salary_usd' in df.columns:
            category_data = df[df[category_col] == category_value]['log_salary_usd']
            if len(category_data) > 0:
                category_avg = np.expm1(category_data.mean())
                diff = predicted_salary - category_avg
                return f"${diff:+,.0f}"
            return "N/A"
        elif 'salary_usd_calculated' in df.columns:
            salary_column = 'salary_usd_calculated'
        
        if salary_column and category_col in df.columns:
            category_data = df[df[category_col] == category_value][salary_column]
            if len(category_data) > 0:
                category_avg = category_data.mean()
                diff = predicted_salary - category_avg
                return f"${diff:+,.0f}"
        return "N/A"
    except Exception as e:
        st.warning(f"Error comparing to category: {e}")
        return "N/A"

def compare_to_hub_dynamic(df, predicted_salary, hub_name):
    """Compare predicted salary to hub average using dynamic approach"""
    try:
        hub_col = f'Hub_{hub_name}'
        
        if hub_col in df.columns:
            if 'salary_usd' in df.columns:
                hub_data = df[df[hub_col] == 1]['salary_usd']
                if len(hub_data) > 0:
                    hub_avg = hub_data.mean()
                    diff = predicted_salary - hub_avg
                    return f"${diff:+,.0f}"
            elif 'log_salary_usd' in df.columns:
                hub_data = df[df[hub_col] == 1]['log_salary_usd']
                if len(hub_data) > 0:
                    hub_avg = np.expm1(hub_data.mean())
                    diff = predicted_salary - hub_avg
                    return f"${diff:+,.0f}"
            elif 'salary_usd_calculated' in df.columns:
                hub_data = df[df[hub_col] == 1]['salary_usd_calculated']
                if len(hub_data) > 0:
                    hub_avg = hub_data.mean()
                    diff = predicted_salary - hub_avg
                    return f"${diff:+,.0f}"
        return "N/A"
    except Exception as e:
        st.warning(f"Error comparing to hub: {e}")
        return "N/A"