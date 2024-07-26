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
with st.sidebar.expander("What is the difference?"):
    st.write("""
    We measure the performance of each model using **Root Mean Square Error (RMSE)**.
    It measures the average magnitude of the errors between predicted values and observed values.

    A value of 0 would indicate a perfect fit to the data.
    The lesser the value of RMSE, the more accurate the model.
    """)

    st.write("### RMSE for Each Model:")
    st.write("""
    - RMSE for XGBoost: 427.71908954105385
    - RMSE for CatBoost: 429.4578969913757
    - RMSE for AdaBoost: 1037.100131525474
    - RMSE for HistGradientBoosting: 432.1493228569663
    """)

    if st.button("Show RMSE Plots"):
        st.image("Data/residual_plots.png", caption="Residual Plots", use_column_width=True)
        st.image("Data/residual_histograms.png", caption="Residual Histograms", use_column_width=True)

# Apply the background color based on the selected model
background_color = background_colors[selected_model_name]
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amatic+SC&family=Gloria+Hallelujah&display=swap');

    .stApp {{
        background-color: {background_color};
    }}
    
    .title {{
        font-family: 'Amatic SC', cursive;
        font-size: 3em;
    }}
    
    .input-label {{
        font-family: 'Gloria Hallelujah', cursive;
        margin-bottom: -10px;
    }}
    
    .selected-model {{
        font-family: 'Amatic SC', cursive;
        font-size: 2em;
        position: absolute;
        top: 10px;
        right: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Show the selected model name in a funky font at the top right
st.markdown(f"<div class='selected-model'>Model: {selected_model_name}</div>", unsafe_allow_html=True)

# Set up the Streamlit app title
st.markdown("<h1 class='title'>Diamond Price Prediction</h1>", unsafe_allow_html=True)

# User inputs for the features required for prediction
st.markdown("<h3 class='input-label'>Carat</h3>", unsafe_allow_html=True)
carat = st.number_input("Carat", min_value=0.1, max_value=5.01, value=0.5, step=0.01, key="carat", label_visibility="hidden")

st.markdown("<h3 class='input-label'>Length</h3>", unsafe_allow_html=True)
x = st.number_input("Length", min_value=0.1, max_value=10.74, value=5.0, step=0.1, key="length", label_visibility="hidden")

st.markdown("<h3 class='input-label'>Breadth</h3>", unsafe_allow_html=True)
y = st.number_input("Breadth", min_value=0.1, max_value=58.9, value=10.0, step=0.1, key="breadth", label_visibility="hidden")

st.markdown("<h3 class='input-label'>Width</h3>", unsafe_allow_html=True)
z = st.number_input("Width", min_value=0.1, max_value=31.8, value=5.0, step=0.1, key="width", label_visibility="hidden")

st.markdown("<h3 class='input-label'>Cut</h3>", unsafe_allow_html=True)
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], key="cut", label_visibility="hidden")

st.markdown("<h3 class='input-label'>Color</h3>", unsafe_allow_html=True)
color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'], key="color", label_visibility="hidden")

st.markdown("<h3 class='input-label'>Clarity</h3>", unsafe_allow_html=True)
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], key="clarity", label_visibility="hidden")

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

# Sidebar for dataset information
with st.sidebar.expander("What should I upload?"):
    st.write("""
    The uploaded dataset must be a CSV file, and must have the following columns in the following order:
                     
    - carat: weight of the diamond
    - cut: quality of the cut (Fair(Worst), Good, Very Good, Premium, Ideal(Bet))
    - color: diamond colour, from J (Worst) to D (Best)
    - clarity: a measurement of how clear the diamond is (I1 (Worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (Best))
    - depth: total depth percentage = z / mean(x, y)
    - table: width of top of diamond relative to widest point
    - x: length in mm
    - y: width in mm
    - z: depth in mm
    - price: price in US dollars
    """)

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
