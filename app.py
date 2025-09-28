# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import os
import warnings
warnings.filterwarnings('ignore')

# ======== ALL PREPROCESSING FUNCTIONS AND PIPELINES ========

def temp_cat(X):
    if isinstance(X, pd.DataFrame):
        X['avg_temp_cat'] = pd.cut(X['avg_temp'], bins=[0, 5, 10, 20, 30, np.inf], labels=['very_cold', 'cold', 'warm', 'hot', 'very_hot'])
        return X
    else:
        X = pd.DataFrame(X)
        X['avg_temp_cat'] = pd.cut(X['avg_temp'], bins=[0, 5, 10, 20, 30, np.inf], labels=['very_cold', 'cold', 'warm', 'hot', 'very_hot'])
        return X

# Create all the transformers and pipelines
temp_cat_transformer = FunctionTransformer(temp_cat)
temp_cat_pipeline = make_pipeline(
    temp_cat_transformer,
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
)

def clean(X):
    if isinstance(X, pd.DataFrame):
        return X.dropna()
    else:
        return pd.DataFrame(X).dropna()

clean_transformer = FunctionTransformer(clean)
clean_pipeline = make_pipeline(clean_transformer, StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
)

def proxy_humidity(X):
    if isinstance(X, pd.DataFrame):
        X["proxy_humidity"] = X["average_rain_fall_mm_per_year"] / (X["avg_temp"] + 1)
        return X
    else:
        X = pd.DataFrame(X)
        X["proxy_humidity"] = X["average_rain_fall_mm_per_year"] / (X["avg_temp"] + 1)
        return X

proxy_humidity_transformer = FunctionTransformer(proxy_humidity)
proxy_humidity_pipeline = make_pipeline(proxy_humidity_transformer, StandardScaler())

square_transformer = FunctionTransformer(np.square)
square_pipeline = make_pipeline(square_transformer, StandardScaler())

log_transformer = FunctionTransformer(np.log1p)
log_pipeline = make_pipeline(log_transformer, StandardScaler())

default_num_pipeline = make_pipeline(StandardScaler())

# Correlation Threshold Selector Class
class CorrelationThresholdSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9, target_threshold=0.0, method="pearson", min_variance=0.0):
        self.threshold = threshold
        self.target_threshold = target_threshold
        self.method = method
        self.min_variance = min_variance

    def fit(self, X, y):
        X_original = X
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        n_features = X_arr.shape[1]
        self.n_features_in_ = n_features

        if hasattr(X_original, "columns"):
            self.feature_names_in_ = np.asarray(X_original.columns)
        else:
            self.feature_names_in_ = np.array([f"f{i}" for i in range(n_features)])

        if n_features <= 1:
            self.features_to_drop_ = np.array([], dtype=int)
            self.selected_features_ = np.arange(n_features, dtype=int)
            return self

        X_df = pd.DataFrame(X_arr, columns=self.feature_names_in_)
        variances = X_df.var(numeric_only=True)
        low_var_mask = variances <= self.min_variance
        low_var_idx = np.where(low_var_mask)[0].tolist()

        corr_mat = X_df.corr(method=self.method).abs().values
        np.fill_diagonal(corr_mat, 0.0)

        y_series = pd.Series(y_arr)
        target_corr_series = X_df.corrwith(y_series, method=self.method).abs().fillna(0.0)
        target_corr = target_corr_series.values

        visited = set()
        drops = set()

        for i in range(n_features):
            if i in visited or i in low_var_idx:
                continue

            correlated_idx = set(np.where(corr_mat[i] > self.threshold)[0].tolist())
            cluster = {i} | correlated_idx
            visited |= cluster

            if len(cluster) == 1:
                continue

            best = max(cluster, key=lambda idx: (target_corr[idx], X_df.iloc[:, idx].var()))

            if self.target_threshold > 0 and target_corr[best] < self.target_threshold:
                drops |= cluster
            else:
                cluster.remove(best)
                drops |= cluster

        drops |= set(low_var_idx)
        self.features_to_drop_ = np.array(sorted(drops), dtype=int)
        retained = sorted(set(range(n_features)) - set(self.features_to_drop_))
        self.selected_features_ = np.array(retained, dtype=int)
        self.selected_feature_names_ = self.feature_names_in_[self.selected_features_].tolist()
        self.dropped_feature_names_ = self.feature_names_in_[self.features_to_drop_].tolist()

        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        X_arr = check_array(X, accept_sparse=False, dtype=np.float64)

        if self.selected_features_.size == 0:
            return np.empty((X_arr.shape[0], 0), dtype=X_arr.dtype)

        sel = np.asarray(self.selected_features_, dtype=int)
        return X_arr[:, sel]

    def inverse_transform(self, X):
        check_is_fitted(self, "selected_features_")
        X_arr = check_array(X, accept_sparse=False, dtype=np.float64)

        n_samples = X_arr.shape[0]
        full = np.zeros((n_samples, self.n_features_in_), dtype=X_arr.dtype)
        full[:, self.selected_features_] = X_arr
        return full

    def get_support(self, indices=False):
        check_is_fitted(self, "selected_features_")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_features_] = True
        return np.where(mask)[0] if indices else mask

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        if input_features is None:
            input_features = self.feature_names_in_
        input_features = np.asarray(input_features)
        if len(input_features) != self.n_features_in_:
            raise ValueError("input_features length mismatch")
        return input_features[self.selected_features_]

# ======== FIXED MODEL LOADING ========

def load_model_properly():
    """Load the actual trained model without fallback bullshit"""
    model_path = 'CropYieldPredictor.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found in current directory!")
        st.error("Please make sure 'CropYieldPredictor.pkl' is in the same folder as this script.")
        return None
    
    try:
        # Try different protocols
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success("✅ Trained model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading trained model: {str(e)}")
        
        # Try alternative loading methods
        try:
            import joblib
            model = joblib.load(model_path)
            st.success("✅ Model loaded with joblib!")
            return model
        except:
            pass
            
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file, encoding='latin1')
            st.success("✅ Model loaded with latin1 encoding!")
            return model
        except Exception as e2:
            st.error(f"❌ All loading methods failed: {str(e2)}")
            return None

# ======== STREAMLIT APP CODE ========

# Page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e8b57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background-color: #f0f8f0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e8b57;
        margin: 20px 0;
    }
    .feature-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff4444;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the actual trained model
@st.cache_resource
def load_model():
    return load_model_properly()

# Available areas
AVAILABLE_AREAS = [
    'Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 
    'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 
    'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada', 
    'Central African Republic', 'Chile', 'Colombia', 'Croatia', 'Denmark', 
    'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea', 'Estonia', 
    'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 
    'Guyana', 'Haiti', 'Honduras', 'Hungary', 'India', 'Indonesia', 'Iraq', 
    'Ireland', 'Italy', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 
    'Lebanon', 'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 
    'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro', 'Morocco', 
    'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 
    'Niger', 'Norway', 'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal', 
    'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia', 
    'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 
    'Switzerland', 'Tajikistan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 
    'Ukraine', 'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe'
]

# Main app
def main():
    st.markdown('<h1 class="main-header">🌾 Crop Yield Predictor | Build BY M Hamza Shahid</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("""
        **Cannot load the trained model. Please check:**
        1. 'CropYieldPredictor.pkl' exists in the current directory
        2. The file is not corrupted
        3. You're using compatible Python/scikit-learn versions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show current directory files
        st.write("**Files in current directory:**")
        current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
        st.write(current_files)
        st.stop()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Input Parameters")
        
        with st.form("prediction_form"):
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            
            # Input fields with dropdown for areas and text input for crops
            area = st.selectbox("🌍 Country/Area", AVAILABLE_AREAS, index=AVAILABLE_AREAS.index('India'))
            item = st.text_input("🌱 Crop Type", "Maize")
            year = st.number_input("📅 Year", min_value=1960, max_value=2030, value=2023)
            rainfall = st.text_input("💧 Average Rainfall (mm/year)", "800.0")
            pesticides = st.text_input("🧴 Pesticides (tonnes)", "5000.0")
            temperature = st.text_input("🌡️ Average Temperature (°C)", "20.0")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Predict Yield", use_container_width=True)
    
    with col2:
        st.subheader("📈 Prediction Results")
        
        if submitted:
            try:
                # Convert text inputs to float
                rainfall_val = float(rainfall)
                pesticides_val = float(pesticides)
                temperature_val = float(temperature)
                
                # Create input data for the model
                input_data = {
                    'Area': [area],
                    'Item': [item],
                    'Year': [year],
                    'average_rain_fall_mm_per_year': [rainfall_val],
                    'pesticides_tonnes': [pesticides_val],
                    'avg_temp': [temperature_val]
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame(input_data)
                
                # Show input data
                st.write("**Input Data:**")
                st.dataframe(input_df, use_container_width=True)
                
                # Make prediction with spinner
                with st.spinner("🤖 Making prediction with trained model..."):
                    prediction = model.predict(input_df)
                    predicted_yield = prediction[0]
                
                # Convert hg/ha to kg/ha
                predicted_yield_kg_ha = predicted_yield * 0.1
                
                # Display results
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.metric("Predicted Yield", f"{predicted_yield_kg_ha:,.0f} kg/ha", 
                         delta=f"{predicted_yield:,.0f} hg/ha")
                
                # Interpretation
                if predicted_yield_kg_ha < 2000:
                    st.warning("📉 Below average yield predicted. Consider optimizing farming practices.")
                elif predicted_yield_kg_ha > 5000:
                    st.success("📈 Excellent yield predicted! Optimal conditions detected.")
                else:
                    st.info("📊 Good yield predicted within normal range.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except ValueError:
                st.error("❌ Please enter valid numeric values for Rainfall, Pesticides, and Temperature")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("This might be a feature name mismatch. Check if your trained model expects the exact same feature names.")
                
                # Debug info
                with st.expander("🔧 Debug Information"):
                    st.write("Model type:", type(model))
                    if hasattr(model, 'feature_names_in_'):
                        st.write("Expected features:", model.feature_names_in_)
                    st.write("Input features:", list(input_df.columns))
    
    # Model information in sidebar
    with st.sidebar:
        st.subheader("ℹ️ Model Information")
        st.write(f"**Model Type:** {type(model).__name__}")
        
        # Show model details
        if hasattr(model, 'steps'):
            st.write("**Pipeline Steps:**")
            for step_name, step in model.steps:
                st.write(f"- {step_name}: {type(step).__name__}")
        
        st.write("**Features Used:**")
        st.write("- Area (Country/Region)")
        st.write("- Item (Crop Type)") 
        st.write("- Year")
        st.write("- average_rain_fall_mm_per_year")
        st.write("- pesticides_tonnes")
        st.write("- avg_temp")
        
        st.subheader("🔧 Model Status")
        st.success("✅ Trained model loaded and ready!")
        
        # File info
        model_path = 'CropYieldPredictor.pkl'
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / 1024 / 1024
            st.write(f"**Model file size:** {file_size:.2f} MB")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit | Build BY M Hamza Shahid | This project is build for Uraan AI Techathton 1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()