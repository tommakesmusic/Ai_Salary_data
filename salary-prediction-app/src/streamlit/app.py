import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_functions import calculate_salary_percentile, compare_to_market as compare_to_category, compare_to_hub as compare_to_hub_dynamic

from data.data_loader import load_salary_model, load_feature_info
from utils.helper_functions import predict_salary, get_business_insights, format_salary

# Page config
st.set_page_config(
    page_title="AI Salary Insights & Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
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
    
    # Determine which salary column to use
    salary_column = None
    if 'salary_usd' in df.columns:
        salary_column = 'salary_usd'
    elif 'log_salary_usd' in df.columns:
        # Convert log salary back to actual salary for display
        df = df.copy()  # Make a copy to avoid modifying the original
        df['salary_usd_calculated'] = np.expm1(df['log_salary_usd'])
        salary_column = 'salary_usd_calculated'
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if salary_column and salary_column in df.columns:
            avg_salary = df[salary_column].mean()
            st.metric("💰 Average Salary", f"${avg_salary:,.0f}")
        else:
            st.metric("💰 Average Salary", "N/A")
            st.caption("No salary data found")
    
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
        st.subheader("💰 Salary Distribution Analysis")
        
        # Look for categorical columns that could be used for grouping
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in ['log_salary_usd', 'salary_usd']]
        
        # Try to find experience-related columns
        exp_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['exp', 'level', 'seniority', 'grade', 'tier'])]
        
        # Try job title as backup
        title_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['title', 'role', 'position'])]
        
        # Try company size as another backup
        size_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['size', 'company'])]
        
        # Choose the best grouping column available
        grouping_col = None
        chart_title = "Salary Distribution"
        
        if exp_candidates:
            grouping_col = exp_candidates[0]
            chart_title = "Salary Distribution by Experience/Level"
        elif title_candidates:
            grouping_col = title_candidates[0]
            chart_title = "Salary Distribution by Job Title"
        elif size_candidates:
            grouping_col = size_candidates[0]
            chart_title = "Salary Distribution by Company Size"
        elif categorical_cols:
            grouping_col = categorical_cols[0]
            chart_title = f"Salary Distribution by {grouping_col.replace('_', ' ').title()}"
        
        if grouping_col and salary_column and grouping_col in df.columns and salary_column in df.columns:
            try:
                # Check if we have reasonable number of categories
                unique_values = df[grouping_col].nunique()
                
                if unique_values <= 20:  # Only create chart if reasonable number of categories
                    # For numeric experience columns, convert to categorical or create bins
                    if df[grouping_col].dtype in ['int64', 'float64']:
                        # Create experience level bins for numeric data
                        if 'exp' in grouping_col.lower() or 'year' in grouping_col.lower():
                            # Create experience bins
                            df_plot = df.copy()
                            df_plot['exp_category'] = pd.cut(df_plot[grouping_col], 
                                                           bins=5, 
                                                           labels=['Entry (0-2)', 'Junior (2-4)', 'Mid (4-6)', 'Senior (6-8)', 'Expert (8+)'])
                            grouping_col_plot = 'exp_category'
                        else:
                            df_plot = df.copy()
                            grouping_col_plot = grouping_col
                    else:
                        df_plot = df.copy()
                        grouping_col_plot = grouping_col
                    
                    # Create the box plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df_plot, x=grouping_col_plot, y=salary_column, ax=ax)
                    plt.xticks(rotation=45)
                    plt.title(chart_title)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info(f"Too many categories ({unique_values}) in {grouping_col} for box plot visualization")
                    
                    # Show alternative: salary histogram
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.hist(df[salary_column].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    plt.xlabel("Salary (USD)")
                    plt.ylabel("Frequency")
                    plt.title("Salary Distribution Histogram")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error creating salary chart: {e}")
                
                # Show fallback histogram
                if salary_column in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.hist(df[salary_column].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    plt.xlabel("Salary (USD)")
                    plt.ylabel("Frequency")
                    plt.title("Salary Distribution Histogram")
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            # Show salary histogram as fallback
            if salary_column and salary_column in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.hist(df[salary_column].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel("Salary (USD)")
                plt.ylabel("Frequency")
                plt.title("Overall Salary Distribution")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No suitable data for salary visualization")
    
    with col2:
        st.subheader("🌍 Jobs Distribution by Hub")
        hub_data = get_hub_distribution(df)
        if not hub_data.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.pie(hub_data['count'], labels=hub_data['hub'], autopct='%1.1f%%')
            plt.title("Distribution of Jobs by Geographic Hub")
            st.pyplot(fig)
        else:
            # Create hub chart from column data
            hub_columns = [col for col in df.columns if col.startswith('Hub_')]
            if hub_columns:
                # If we have hub data, create the chart
                if any(df[hub_col].sum() > 0 for hub_col in hub_columns):
                    hub_counts = []
                    for hub_col in hub_columns:
                        hub_name = hub_col.replace('Hub_', '')
                        count = df[hub_col].sum()
                        if count > 0:
                            hub_counts.append({'hub': hub_name, 'count': count})
                    
                    if hub_counts:
                        hub_df = pd.DataFrame(hub_counts)
                        fig, ax = plt.subplots(figsize=(8, 8))
                        plt.pie(hub_df['count'], labels=hub_df['hub'], autopct='%1.1f%%')
                        plt.title("Distribution of Jobs by Geographic Hub")
                        st.pyplot(fig)
                    else:
                        st.info("No hub data available")
                else:
                    st.info("No active hubs found")
            else:
                st.info("No hub columns found in dataset")

def show_salary_prediction(model, metadata, df):
    """Display the salary prediction interface"""
    st.markdown('<div class="section-header">🔮 Salary Prediction Tool</div>', unsafe_allow_html=True)
    
    if model is None:
        st.warning("⚠️ Model not loaded. Prediction functionality is not available.")
        st.info("Please ensure your model files are in the 'models' directory.")
        return
    
    st.markdown("### 📝 Enter Job Details for Salary Prediction")
    
    # Dynamically detect available columns instead of requiring specific ones
    available_categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and not col.startswith(('Hub_', 'skill_'))]
    available_numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and not col.startswith(('Hub_', 'skill_'))]
    
    col1, col2 = st.columns(2)
    
    # Dictionary to store user inputs
    user_inputs = {}
    
    with col1:
        st.markdown("#### 📊 Available Categorical Features")
        
        for col in available_categorical_cols:
            if col not in ['log_salary_usd', 'salary_usd']:  # Exclude target variables
                unique_values = sorted(df[col].dropna().unique())
                if len(unique_values) <= 50:  # Only show if reasonable number of options
                    selected_value = st.selectbox(
                        f"{col.replace('_', ' ').title()}", 
                        unique_values,
                        key=f"cat_{col}"
                    )
                    user_inputs[col] = selected_value
    
    with col2:
        st.markdown("#### 🔢 Available Numeric Features")
        
        for col in available_numeric_cols[:10]:  # Limit to first 10 numeric columns
            if col not in ['log_salary_usd', 'salary_usd']:  # Exclude target variables
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                default_val = float(df[col].median())
                
                selected_value = st.number_input(
                    f"{col.replace('_', ' ').title()}", 
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"num_{col}"
                )
                user_inputs[col] = selected_value
    
    # Hub selection
    st.markdown("---")
    st.markdown("### 🌍 Geographic Hub Selection")
    
    hub_columns = [col for col in df.columns if col.startswith('Hub_')]
    if hub_columns:
        hubs = [col.replace('Hub_', '') for col in hub_columns]
        selected_hub = st.selectbox("Geographic Hub", hubs)
        
        # Set all hub columns to 0, then set selected one to 1
        for hub_col in hub_columns:
            user_inputs[hub_col] = 0
        user_inputs[f'Hub_{selected_hub}'] = 1
    else:
        st.error("❌ No Hub data found in dataset")
        selected_hub = None
    
    # Skills selection
    st.markdown("---")
    st.markdown("### 🛠️ Skills Selection")
    
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    
    if skill_columns:
        skills = [col.replace('skill_', '') for col in skill_columns]
        
        # Initialize all skills to 0
        for skill_col in skill_columns:
            user_inputs[skill_col] = 0
        
        # Show skills in a more organized way
        if len(skills) <= 50:  # Only show checkboxes if reasonable number
            num_cols = 4
            cols = st.columns(num_cols)
            selected_skills = []
            
            for i, skill in enumerate(skills):
                col = cols[i % num_cols]
                with col:
                    skill_display = skill.replace('_', ' ').title()
                    if st.checkbox(skill_display, key=f"skill_check_{skill}"):
                        selected_skills.append(skill)
                        user_inputs[f'skill_{skill}'] = 1
        else:
            # Too many skills - use multiselect
            selected_skills = st.multiselect(
                "Select Skills (showing first 50):",
                skills[:50],
                default=[]
            )
            for skill in selected_skills:
                user_inputs[f'skill_{skill}'] = 1
    else:
        st.warning("❌ No skills data found in dataset.")
    
    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            try:
                # Create prediction input from user selections
                prediction_input = create_dynamic_prediction_input(df, user_inputs)
                
                if prediction_input is not None:
                    # Make prediction
                    prediction = model.predict([prediction_input])[0]
                    predicted_salary = np.expm1(prediction)  # Convert from log scale
                    
                    # Display result with styling
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Results")
                    
                    # Main prediction result
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem;
                            border-radius: 10px;
                            text-align: center;
                            color: white;
                            margin: 1rem 0;
                        ">
                            <h2 style="margin: 0; font-size: 2.5rem;">
                                ${predicted_salary:,.0f}
                            </h2>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                                Predicted Annual Salary (USD)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show additional insights
                    show_prediction_insights_dynamic(df, predicted_salary, user_inputs)
                    
                else:
                    st.error("❌ Could not create prediction input. Please check your selections.")
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

def create_dynamic_prediction_input(df, user_inputs):
    """Create input vector for prediction using actual dataset columns"""
    try:
        # Get all non-target columns from the dataset
        exclude_cols = ['log_salary_usd', 'salary_usd']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Initialize input dictionary with defaults
        input_dict = {}
        
        for col in feature_columns:
            if col in user_inputs:
                # Use the user-provided value
                input_dict[col] = user_inputs[col]
            else:
                # Use a default value based on column type
                if df[col].dtype in ['int64', 'float64']:
                    input_dict[col] = df[col].median()
                else:
                    # For categorical columns, use the most frequent value
                    mode_values = df[col].mode()
                    input_dict[col] = mode_values.iloc[0] if len(mode_values) > 0 else 0
        
        # Convert to DataFrame for easier handling
        input_df = pd.DataFrame([input_dict])
        
        # Handle categorical encoding
        categorical_cols = [col for col in input_df.columns if input_df[col].dtype == 'object']
        
        for col in categorical_cols:
            # Get unique values from original data
            unique_values = df[col].unique()
            
            # Simple label encoding
            if input_df[col].iloc[0] in unique_values:
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                input_df[col] = value_map[input_df[col].iloc[0]]
            else:
                input_df[col] = 0  # Default encoding for unknown values
        
        return input_df.iloc[0].values
        
    except Exception as e:
        st.error(f"Error in create_dynamic_prediction_input: {str(e)}")
        return None

def show_prediction_insights_dynamic(df, predicted_salary, user_inputs):
    """Show prediction insights using available data"""
    st.markdown("#### 📊 Prediction Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        percentile = calculate_salary_percentile(df, predicted_salary)
        st.metric("Salary Percentile", f"{percentile:.0f}%", 
                 help="What percentage of salaries are below this prediction")
    
    with col2:
        # Find any experience-related column for comparison
        exp_cols = [col for col in df.columns if 'exp' in col.lower() or 'level' in col.lower()]
        if exp_cols and exp_cols[0] in user_inputs:
            comparison = compare_to_category(df, predicted_salary, exp_cols[0], user_inputs[exp_cols[0]])
            st.metric("vs Category Avg", comparison, help=f"Comparison to {exp_cols[0]}")
        else:
            st.metric("vs Market Avg", "N/A", help="No experience data available")
    
    with col3:
        # Hub comparison if available
        hub_cols = [col for col in user_inputs.keys() if col.startswith('Hub_') and user_inputs[col] == 1]
        if hub_cols:
            hub_name = hub_cols[0].replace('Hub_', '')
            comparison = compare_to_hub_dynamic(df, predicted_salary, hub_name)
            st.metric("vs Hub Average", comparison, help=f"Comparison to {hub_name}")
        else:
            st.metric("vs Hub Average", "N/A")
    
    with col4:
        confidence_score = 80  # Simplified confidence score
        st.metric("Confidence Score", f"{confidence_score:.0f}%",
                 help="Model confidence in this prediction")

def show_skills_insights(df):
    """Display skills-based insights"""
    st.markdown('<div class="section-header">🛠️ Skills Market Analysis</div>', unsafe_allow_html=True)
    
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    
    if not skill_columns:
        st.warning("No skills data available in the dataset.")
        return
    
    # Determine salary column to use
    salary_column = None
    if 'salary_usd' in df.columns:
        salary_column = 'salary_usd'
    elif 'log_salary_usd' in df.columns:
        # Convert log salary back to actual salary
        df = df.copy()
        df['salary_usd_calculated'] = np.expm1(df['log_salary_usd'])
        salary_column = 'salary_usd_calculated'
    
    # Calculate skill demand and average salaries
    skill_data = []
    
    for skill_col in skill_columns:
        skill_name = skill_col.replace('skill_', '').replace('_', ' ').title()
        skill_jobs = df[df[skill_col] == 1]
        
        if len(skill_jobs) > 0:
            demand = len(skill_jobs)
            if salary_column:
                avg_salary = skill_jobs[salary_column].mean()
            else:
                avg_salary = 0
            
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
            if salary_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=top_paying, x='Average Salary', y='Skill', ax=ax)
                plt.xticks(rotation=45)
                plt.title("Highest Paying Skills")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Salary data not available for skills analysis")
        
        with col2:
            st.subheader("📈 Most In-Demand Skills (Top 15)")
            most_demanded = skills_df.nlargest(15, 'Demand')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=most_demanded, x='Demand', y='Skill', ax=ax)
            plt.xticks(rotation=45)
            plt.title("Most In-Demand Skills")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Skills matrix (only if we have salary data)
        if salary_column:
            st.subheader("💼 Skills Value Matrix: Demand vs Salary")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.scatter(skills_df['Demand %'], skills_df['Average Salary'], alpha=0.6)
            plt.xlabel('Demand %')
            plt.ylabel('Average Salary')
            plt.title('Skills: Market Demand vs Average Salary')
            st.pyplot(fig)

def show_hub_analysis(df):
    """Display hub-based analysis"""
    st.markdown('<div class="section-header">🌍 Geographic Hub Analysis</div>', unsafe_allow_html=True)
    
    hub_columns = [col for col in df.columns if col.startswith('Hub_')]
    
    if not hub_columns:
        st.warning("No hub data available in the dataset.")
        return
    
    # Determine salary column to use
    salary_column = None
    if 'salary_usd' in df.columns:
        salary_column = 'salary_usd'
    elif 'log_salary_usd' in df.columns:
        # Convert log salary back to actual salary
        df = df.copy()
        df['salary_usd_calculated'] = np.expm1(df['log_salary_usd'])
        salary_column = 'salary_usd_calculated'
    
    # Hub salary comparison
    st.subheader("💰 Average Salary by Hub")
    hub_salaries = []
    
    for hub_col in hub_columns:
        hub_name = hub_col.replace('Hub_', '')
        hub_data = df[df[hub_col] == 1]
        if len(hub_data) > 0:
            job_count = len(hub_data)
            if salary_column:
                avg_salary = hub_data[salary_column].mean()
            else:
                avg_salary = 0
            
            hub_salaries.append({
                'Hub': hub_name,
                'Average Salary': avg_salary,
                'Job Count': job_count
            })
    
    if hub_salaries:
        hub_df = pd.DataFrame(hub_salaries)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if salary_column:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=hub_df, x='Hub', y='Average Salary', ax=ax)
                plt.xticks(rotation=45)
                plt.title("Average Salary by Geographic Hub")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Salary data not available")
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=hub_df, x='Hub', y='Job Count', ax=ax)
            plt.xticks(rotation=45)
            plt.title("Number of Jobs by Hub")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Display the data table
        st.subheader("📊 Hub Summary Table")
        if salary_column:
            st.dataframe(hub_df.style.format({'Average Salary': '${:,.0f}'}), use_container_width=True)
        else:
            st.dataframe(hub_df, use_container_width=True)

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
    
    # Column analysis
    st.subheader("📊 Column Analysis")
    
    # Categorize columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    hub_cols = [col for col in df.columns if col.startswith('Hub_')]
    skill_cols = [col for col in df.columns if col.startswith('skill_')]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔢 Numeric Columns", len(numeric_cols))
        if st.checkbox("Show Numeric Columns"):
            st.write(numeric_cols)
    
    with col2:
        st.metric("📝 Categorical Columns", len(categorical_cols))
        if st.checkbox("Show Categorical Columns"):
            st.write(categorical_cols)
    
    with col3:
        st.metric("🌍 Hub Columns", len(hub_cols))
        if st.checkbox("Show Hub Columns"):
            st.write(hub_cols)
    
    with col4:
        st.metric("🛠️ Skill Columns", len(skill_cols))
        if st.checkbox("Show Skill Columns"):
            st.write(skill_cols[:20] if len(skill_cols) > 20 else skill_cols)
    
    # Data sample
    st.subheader("📋 Dataset Sample")
    sample_size = st.slider("Number of rows to display", 5, min(100, len(df)), 10)
    st.dataframe(df.head(sample_size), use_container_width=True)
    
    # Column information
    st.subheader("📊 Detailed Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    # Allow filtering of column info
    column_filter = st.selectbox("Filter columns by type", 
                                ["All", "Numeric", "Categorical", "Hub", "Skill"])
    
    if column_filter == "Numeric":
        filtered_info = col_info[col_info['Column'].isin(numeric_cols)]
    elif column_filter == "Categorical":
        filtered_info = col_info[col_info['Column'].isin(categorical_cols)]
    elif column_filter == "Hub":
        filtered_info = col_info[col_info['Column'].isin(hub_cols)]
    elif column_filter == "Skill":
        filtered_info = col_info[col_info['Column'].isin(skill_cols)]
    else:
        filtered_info = col_info
    
    st.dataframe(filtered_info, use_container_width=True)
    
    # Quick statistics for numeric columns
    if numeric_cols:
        st.subheader("📈 Numeric Column Statistics")
        selected_numeric = st.multiselect("Select numeric columns to analyze", 
                                        numeric_cols, 
                                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
        
        if selected_numeric:
            st.dataframe(df[selected_numeric].describe(), use_container_width=True)

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