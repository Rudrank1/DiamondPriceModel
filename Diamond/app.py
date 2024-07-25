import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from user_model import train_user_model

# Function to get or initialize session state
def get_session_state():
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        return False
    return True

# Delete user model from previous sessions if it exists and this is a new session
if not get_session_state():
    user_model_path = 'user_model.pkl'
    if os.path.exists(user_model_path):
        os.remove(user_model_path)

# Define background colors for each model
background_colors = {
    'XGBoost': '#692e2e',
    'CatBoost': '#692e4b',
    'AdaBoost': '#2e4b69',
    'HistGradientBoosting': '#0d4a13',
    'UserModel': '#2e6b2e'
}

# Load all pre-trained models
models = {
    'XGBoost': joblib.load('models/XGBoost_model.pkl'),
    'CatBoost': joblib.load('models/CatBoost_model.pkl'),
    'AdaBoost': joblib.load('models/AdaBoost_model.pkl'),
    'HistGradientBoosting': joblib.load('models/HistGradientBoosting_model.pkl')
}

# Sidebar for model selection
st.sidebar.title("Model Selection and Info")
selected_model_name = st.sidebar.radio("Select Model", list(models.keys()) + ['UserModel'])
selected_model = models.get(selected_model_name)

# Info tab for RMSE
with st.sidebar.expander("What is RMSE?"):
    st.write("""
    **Root Mean Square Error (RMSE)** is a standard way to measure the error of a model in predicting quantitative data. 
    It measures the average magnitude of the errors between predicted values and observed values.
    The formula for RMSE is:

    RMSE = sqrt( (1/n) * Σ(actual - predicted)^2 )

    The RMSE value is always non-negative, and a value of 0 would indicate a perfect fit to the data.
    """)

    st.write("### RMSE for Each Model:")
    st.write("""
    - RMSE for LightGBM: 423.15988213338244
    - RMSE for XGBoost: 427.71908954105385
    - RMSE for CatBoost: 429.4578969913757
    - RMSE for RandomForest: 469.34313557062336
    - RMSE for AdaBoost: 1037.100131525474
    - RMSE for ExtraTrees: 478.55164915608935
    - RMSE for HistGradientBoosting: 432.1493228569663
    """)

# Apply the background color based on the selected model
background_color = background_colors[selected_model_name]
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Set up the Streamlit app title
st.title("Diamond Price Prediction")

# User inputs for the features required for prediction
carat = st.number_input("Carat", min_value=0.1, max_value=5.01, value=0.5, step=0.01)
x = st.number_input("Length", min_value=0.1, max_value=10.74, value=5.0, step=0.1)
y = st.number_input("Breadth", min_value=0.1, max_value=58.9, value=10.0, step=0.1)
z = st.number_input("Width", min_value=0.1, max_value=31.8, value=5.0, step=0.1)
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Create a DataFrame from the user input
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity],
    'x': [x],
    'y': [y],
    'z': [z]
})

# Correcting the skewness of carat, length and width
input_data['carat'] = np.log(input_data['carat'])
input_data['x'] = np.log(input_data['x'])
input_data['y'] = np.log(input_data['y'])

# Mappings to convert categorical features to numerical
cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

# Apply the mappings to the input data
input_data['cut'] = input_data['cut'].map(cut_mapping)
input_data['color'] = input_data['color'].map(color_mapping)
input_data['clarity'] = input_data['clarity'].map(clarity_mapping)

# Predict the price when the button is clicked
if st.button("Predict"):
    if selected_model_name == 'UserModel':
        try:
            selected_model = joblib.load("user_model.pkl")
        except FileNotFoundError:
            st.write("Please train your model first!")
            selected_model = None

    if selected_model:
        prediction = selected_model.predict(input_data)
        st.write(f"Predicted Price using {selected_model_name}: ${prediction[0]:.2f}")

# Sidebar for training user model
st.sidebar.title("Train Your Own Model")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
model_type = st.sidebar.selectbox("Select model to train", list(models.keys()))
train_button = st.sidebar.button("Train Model")

if uploaded_file and train_button:
    data = pd.read_csv(uploaded_file)
    st.sidebar.write("Dataset uploaded successfully!")
    
    # Train the selected model
    user_model = train_user_model(data, model_type)
    st.sidebar.success("Model trained and saved as user_model.pkl")
