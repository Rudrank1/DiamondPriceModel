import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib

# Load the pre-trained model
model = joblib.load('xgboost_model.pkl')

# Set up the Streamlit app title
st.title("Diamond Price Prediction")

# User inputs for the features required for prediction
carat = st.number_input("Carat", min_value=0.2, max_value=5.01, value=0.5, step=0.01)
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Create a DataFrame from the user input
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity]
})

# Correcting the skewness of carat
input_data['carat'] = np.log(input_data['carat'])

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
    prediction = model.predict(input_data)
    st.write(f"Predicted Price: ${prediction[0]:.2f}")
