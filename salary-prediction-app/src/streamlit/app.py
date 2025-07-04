import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_salary_model, load_feature_info
from utils.helper_functions import predict_salary, get_business_insights, format_salary

# Page config
st.set_page_config(
    page_title="AI Salary Insights & Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #3498db;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Now use the correct path relative to project root
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ai_job_dataset_cleaned.csv')
        df = pd.read_csv(data_path)
        st.success(f"✅ Dataset loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info(f"Looking for data at: {data_path}")
        return None

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        # Use the correct path to models directory
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            if model_files:
                latest_model = sorted(model_files)[-1]  # Get the most recent
                model_path = os.path.join(models_dir, latest_model)
                model, metadata = load_salary_model(model_path)
                if model is not None:
                    st.success(f"✅ Model loaded successfully: {latest_model}")
                    return model, metadata
        
        st.error("❌ Model not found in models directory")
        st.info(f"Looking for models in: {models_dir}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 AI Salary Insights & Prediction Platform</h1>', unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    model, metadata = load_model()
    
    if df is None:
        st.error("❌ Cannot proceed without data. Please check your data file.")
        return
    
    # Show data loading status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Data Status", "✅ Loaded" if df is not None else "❌ Failed")
    with col2:
        st.metric("🤖 Model Status", "✅ Ready" if model is not None else "❌ Not Available")
    with col3:
        st.metric("📈 Records", f"{len(df):,}" if df is not None else "0")
    
    # Sidebar for navigation and filters
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        
        section = st.radio(
            "Choose Section:",
            ["📊 Dashboard Overview", "🔮 Salary Prediction", "🌍 Hub Analysis", "🛠️ Skills Insights", "📈 Data Explorer"]
        )
        
        st.markdown("---")
        st.markdown("### 🎛️ Global Filters")
        
        # Get unique values for filters
        hub_columns = [col for col in df.columns if col.startswith('Hub_')]
        hubs = []
        for hub_col in hub_columns:
            if df[hub_col].sum() > 0:  # Only include hubs with data
                hubs.append(hub_col.replace('Hub_', ''))
        
        selected_hubs = st.multiselect("Select Hubs:", hubs, default=hubs[:3] if len(hubs) > 3 else hubs)
        
        # Experience level filter
        if 'experience_level' in df.columns:
            exp_levels = df['experience_level'].unique()
            selected_exp = st.multiselect("Experience Level:", exp_levels, default=list(exp_levels))
        else:
            selected_exp = []
        
        # Company size filter
        if 'company_size' in df.columns:
            company_sizes = df['company_size'].unique()
            selected_company_size = st.multiselect("Company Size:", company_sizes, default=list(company_sizes))
        else:
            selected_company_size = []

    # Filter data based on sidebar selections
    filtered_df = df.copy()
    if selected_hubs:
        hub_filter = pd.Series([False] * len(filtered_df))
        for hub in selected_hubs:
            hub_col = f'Hub_{hub}'
            if hub_col in filtered_df.columns:
                hub_filter |= (filtered_df[hub_col] == 1)
        filtered_df = filtered_df[hub_filter]
    
    if selected_exp and 'experience_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['experience_level'].isin(selected_exp)]
    
    if selected_company_size and 'company_size' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['company_size'].isin(selected_company_size)]

    # Main content based on selected section
    if section == "📊 Dashboard Overview":
        show_dashboard_overview(filtered_df)
    elif section == "🔮 Salary Prediction":
        show_salary_prediction(model, metadata, df)
    elif section == "🌍 Hub Analysis":
        show_hub_analysis(filtered_df)
    elif section == "🛠️ Skills Insights":
        show_skills_insights(filtered_df)
    elif section == "📈 Data Explorer":
        show_data_explorer(filtered_df)

def show_dashboard_overview(df):
    """Display the main dashboard with key metrics and insights"""
    st.markdown('<div class="section-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'salary_usd' in df.columns:
            avg_salary = df['salary_usd'].mean()
            st.metric("💰 Average Salary", f"${avg_salary:,.0f}")
        else:
            st.metric("💰 Average Salary", "N/A")
    
    with col2:
        total_jobs = len(df)
        st.metric("📊 Total Jobs", f"{total_jobs:,}")
    
    with col3:
        if 'job_title' in df.columns:
            unique_roles = df['job_title'].nunique()
            st.metric("💼 Unique Roles", unique_roles)
        else:
            st.metric("💼 Unique Roles", "N/A")
    
    with col4:
        hub_columns = [col for col in df.columns if col.startswith('Hub_')]
        active_hubs = len([col for col in hub_columns if df[col].sum() > 0])
        st.metric("🌍 Active Hubs", active_hubs)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Salary Distribution by Experience Level")
        if 'experience_level' in df.columns and 'salary_usd' in df.columns:
            fig = px.box(df, x='experience_level', y='salary_usd', 
                        title="Salary Distribution by Experience Level",
                        color='experience_level')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Salary or experience level data not available")
    
    with col2:
        st.subheader("🌍 Jobs Distribution by Hub")
        hub_data = get_hub_distribution(df)
        if not hub_data.empty:
            fig = px.pie(hub_data, values='count', names='hub', 
                        title="Distribution of Jobs by Geographic Hub",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hub distribution data not available")

def show_salary_prediction(model, metadata, df):
    """Display the salary prediction interface"""
    st.markdown('<div class="section-header">🔮 Salary Prediction Tool</div>', unsafe_allow_html=True)
    
    if model is None:
        st.warning("⚠️ Model not loaded. Prediction functionality is not available.")
        st.info("Please ensure your model files are in the 'models' directory.")
        return
    
    st.markdown("### Enter Job Details for Salary Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Experience level
        exp_levels = df['experience_level'].unique() if 'experience_level' in df.columns else ['EN', 'MI', 'SE', 'EX']
        experience_level = st.selectbox("Experience Level", exp_levels)
        
        # Employment type
        emp_types = df['employment_type'].unique() if 'employment_type' in df.columns else ['FT', 'PT', 'CT', 'FL']
        employment_type = st.selectbox("Employment Type", emp_types)
        
        # Company size
        company_sizes = df['company_size'].unique() if 'company_size' in df.columns else ['S', 'M', 'L']
        company_size = st.selectbox("Company Size", company_sizes)
    
    with col2:
        # Hub selection
        hub_columns = [col for col in df.columns if col.startswith('Hub_')]
        hubs = [col.replace('Hub_', '') for col in hub_columns]
        selected_hub = st.selectbox("Geographic Hub", hubs)
        
        # Job title
        job_titles = df['job_title'].unique() if 'job_title' in df.columns else ['Data Scientist']
        job_title = st.selectbox("Job Title", sorted(job_titles))
        
        # Remote ratio
        remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, 50)
    
    # Skills selection
    st.markdown("### Select Relevant Skills")
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    skills = [col.replace('skill_', '') for col in skill_columns]
    
    if skills:
        col1, col2, col3 = st.columns(3)
        selected_skills = []
        
        for i, skill in enumerate(skills):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.checkbox(skill.replace('_', ' ').title(), key=f"skill_{skill}"):
                    selected_skills.append(skill)
        
        # Prediction button
        if st.button("🔮 Predict Salary", type="primary"):
            try:
                # Create input vector (simplified version)
                st.info("🔧 Prediction feature coming soon! Model loaded successfully.")
                st.markdown("**Selected Parameters:**")
                st.write(f"- Experience Level: {experience_level}")
                st.write(f"- Hub: {selected_hub}")
                st.write(f"- Job Title: {job_title}")
                st.write(f"- Skills: {', '.join(selected_skills) if selected_skills else 'None'}")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
    else:
        st.warning("No skills data found in dataset.")

def show_hub_analysis(df):
    """Display hub-based analysis"""
    st.markdown('<div class="section-header">🌍 Geographic Hub Analysis</div>', unsafe_allow_html=True)
    
    hub_columns = [col for col in df.columns if col.startswith('Hub_')]
    
    if not hub_columns:
        st.warning("No hub data available in the dataset.")
        return
    
    # Hub salary comparison
    st.subheader("💰 Average Salary by Hub")
    hub_salaries = []
    
    for hub_col in hub_columns:
        hub_name = hub_col.replace('Hub_', '')
        hub_data = df[df[hub_col] == 1]
        if len(hub_data) > 0 and 'salary_usd' in df.columns:
            avg_salary = hub_data['salary_usd'].mean()
            job_count = len(hub_data)
            hub_salaries.append({
                'Hub': hub_name,
                'Average Salary': avg_salary,
                'Job Count': job_count
            })
    
    if hub_salaries:
        hub_df = pd.DataFrame(hub_salaries)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(hub_df, x='Hub', y='Average Salary', 
                        title="Average Salary by Geographic Hub",
                        color='Average Salary', 
                        color_continuous_scale='viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(hub_df, x='Hub', y='Job Count', 
                        title="Number of Jobs by Hub",
                        color='Job Count', 
                        color_continuous_scale='blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display the data table
        st.subheader("📊 Hub Summary Table")
        st.dataframe(hub_df.style.format({'Average Salary': '${:,.0f}'}), use_container_width=True)

def show_skills_insights(df):
    """Display skills-based insights"""
    st.markdown('<div class="section-header">🛠️ Skills Market Analysis</div>', unsafe_allow_html=True)
    
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    
    if not skill_columns:
        st.warning("No skills data available in the dataset.")
        return
    
    # Calculate skill demand and average salaries
    skill_data = []
    
    for skill_col in skill_columns:
        skill_name = skill_col.replace('skill_', '').replace('_', ' ').title()
        skill_jobs = df[df[skill_col] == 1]
        
        if len(skill_jobs) > 0:
            demand = len(skill_jobs)
            avg_salary = skill_jobs['salary_usd'].mean() if 'salary_usd' in df.columns else 0
            skill_data.append({
                'Skill': skill_name,
                'Demand': demand,
                'Average Salary': avg_salary,
                'Demand %': (demand / len(df)) * 100
            })
    
    if skill_data:
        skills_df = pd.DataFrame(skill_data).sort_values('Average Salary', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Highest Paying Skills (Top 15)")
            top_paying = skills_df.head(15)
            fig = px.bar(top_paying, x='Average Salary', y='Skill', 
                        orientation='h', title="Highest Paying Skills",
                        color='Average Salary', color_continuous_scale='viridis')
            fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Most In-Demand Skills (Top 15)")
            most_demanded = skills_df.nlargest(15, 'Demand')
            fig = px.bar(most_demanded, x='Demand', y='Skill', 
                        orientation='h', title="Most In-Demand Skills",
                        color='Demand', color_continuous_scale='blues')
            fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills matrix
        st.subheader("💼 Skills Value Matrix: Demand vs Salary")
        fig = px.scatter(skills_df, x='Demand %', y='Average Salary', 
                        hover_name='Skill',
                        title="Skills: Market Demand vs Average Salary",
                        size='Demand', 
                        color='Average Salary',
                        color_continuous_scale='plasma')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(df):
    """Display data exploration interface"""
    st.markdown('<div class="section-header">📈 Data Explorer</div>', unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Records", f"{len(df):,}")
    with col2:
        st.metric("🔢 Total Features", len(df.columns))
    with col3:
        missing_data = df.isnull().sum().sum()
        st.metric("❓ Missing Values", missing_data)
    
    # Data table
    st.subheader("📋 Dataset Sample")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Column information
    st.subheader("📊 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(col_info, use_container_width=True)

# Helper functions
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

if __name__ == "__main__":
    main()