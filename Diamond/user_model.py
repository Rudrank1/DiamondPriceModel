import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocessing
from modeling import train_lightgbm, train_xgboost, train_catboost, train_adaboost, train_histgradientboosting

def train_user_model(data, model_type):
    # Preprocess the data
    data = preprocessing.initial_preprocess(data)
    data = preprocessing.encode_categorical(data)
    data = preprocessing.correct_positive_skewness(data, ['carat', 'x', 'y'])
    
    # Define features and target
    features = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z']
    target = 'price'
    
    if target not in data.columns:
        raise ValueError(f"Dataset must contain the target column '{target}'")

    X = data[features]
    y = data[target]
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model based on the chosen type
    if model_type == 'LightGBM':
        model = train_lightgbm(X_train, y_train)
    elif model_type == 'XGBoost':
        model = train_xgboost(X_train, y_train)
    elif model_type == 'CatBoost':
        model = train_catboost(X_train, y_train)
    elif model_type == 'AdaBoost':
        model = train_adaboost(X_train, y_train)
    elif model_type == 'HistGradientBoosting':
        model = train_histgradientboosting(X_train, y_train)
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")
    
    # Save the trained model
    joblib.dump(model, 'models/user_model.pkl')
