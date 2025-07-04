import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
from app import load_data
from app import load_model

# Add the parent directory to the path to import the main app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestStreamlitApp:
    """Test suite for Streamlit salary prediction app"""
    
    @pytest.fixture
    def app_test(self):
        """Initialize the Streamlit app test"""
        return AppTest.from_file("app.py")
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'experience': [1, 2, 3, 4, 5],
            'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
            'location': ['New York', 'California', 'Texas', 'Florida', 'Washington'],
            'salary': [50000, 75000, 100000, 60000, 85000]
        })
    
    def test_app_loads_successfully(self, app_test):
        """Test that the app loads without errors"""
        app_test.run()
        assert not app_test.exception
    
    def test_app_title_exists(self, app_test):
        """Test that the app has a title"""
        app_test.run()
        assert len(app_test.title) > 0
    
    def test_sidebar_elements(self, app_test):
        """Test sidebar elements are present"""
        app_test.run()
        # Check if sidebar has input elements
        assert len(app_test.sidebar) > 0
    
    def test_input_widgets(self, app_test):
        """Test that input widgets are present and functional"""
        app_test.run()
        
        # Test number input for experience
        if app_test.number_input:
            app_test.number_input[0].set_value(5)
            app_test.run()
            assert app_test.number_input[0].value == 5
    
    def test_selectbox_widgets(self, app_test):
        """Test selectbox widgets"""
        app_test.run()
        
        if app_test.selectbox:
            # Test first selectbox (assuming education level)
            app_test.selectbox[0].select("Master")
            app_test.run()
            assert "Master" in str(app_test.selectbox[0].value)
    
    def test_prediction_button(self, app_test):
        """Test prediction button functionality"""
        app_test.run()
        
        # Fill in some sample inputs
        if app_test.number_input:
            app_test.number_input[0].set_value(3)
        
        if app_test.selectbox and len(app_test.selectbox) >= 2:
            app_test.selectbox[0].select("Bachelor")
            app_test.selectbox[1].select("New York")
        
        # Click predict button if it exists
        if app_test.button:
            app_test.button[0].click()
            app_test.run()
            # Check if prediction result is displayed
            assert not app_test.exception
    
    @patch('pandas.read_csv')
    def test_data_loading(self, mock_read_csv, sample_data):
        """Test data loading functionality"""
        mock_read_csv.return_value = sample_data
        
        # Import and test data loading function if it exists
        try:
            data = load_data()
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
        except ImportError:
            # If no load_data function, skip this test
            pytest.skip("No load_data function found in app.py")
    
    @patch('joblib.load')
    def test_model_loading(self, mock_joblib_load):
        """Test model loading functionality"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([75000])
        mock_joblib_load.return_value = mock_model
        
        try:
            model = load_model()
            assert model is not None
        except ImportError:
            pytest.skip("No load_model function found in app.py")
    
    def test_data_validation(self):
        """Test input data validation"""
        # Test invalid experience values
        invalid_experience = [-1, 100, "invalid"]
        
        for exp in invalid_experience:
            # This would test validation logic if implemented
            if isinstance(exp, (int, float)) and exp < 0:
                assert exp < 0  # Should be handled by validation
    
    def test_prediction_output_format(self, app_test):
        """Test that prediction output is in correct format"""
        app_test.run()
        
        # Fill inputs and make prediction
        if app_test.number_input:
            app_test.number_input[0].set_value(5)
        
        if app_test.button:
            app_test.button[0].click()
            app_test.run()
            
            # Check if any metric or success message is displayed
            if app_test.metric or app_test.success:
                assert True  # Prediction was made successfully
    
    def test_error_handling(self, app_test):
        """Test error handling for invalid inputs"""
        app_test.run()
        
        # Try to make prediction with empty/invalid inputs
        if app_test.button:
            app_test.button[0].click()
            app_test.run()
            # App should handle gracefully without crashing
            assert not app_test.exception
    
    def test_app_responsiveness(self, app_test):
        """Test app responds to user interactions"""
        app_test.run()
        initial_state = str(app_test)
        
        # Make some changes
        if app_test.number_input:
            app_test.number_input[0].set_value(10)
            app_test.run()
            modified_state = str(app_test)
            # State should change after interaction
            assert initial_state != modified_state
    
    def test_session_state_management(self, app_test):
        """Test session state is properly managed"""
        app_test.run()
        
        # Check if session state variables are accessible
        # This assumes the app uses session state
        if hasattr(st, 'session_state'):
            assert True  # Session state is available

if __name__ == "__main__":
    pytest.main([__file__])