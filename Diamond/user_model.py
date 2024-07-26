"""
Importing necessary packages
"""
import joblib
from sklearn.model_selection import train_test_split
import preprocessing
from modeling import train_xgboost, train_catboost, train_adaboost, train_histgradientboosting

def train_user_model(data, model_type):
    """
    Train a user-selected model on the provided dataset and save the trained model.

    This function preprocesses the input data, splits it into training and test sets, 
    trains the specified model using the training data, and saves the trained model 
    to the 'models' directory.

    Args:
        data (pd.DataFrame): The dataset to train the model on. Must contain features and target columns.
        model_type (str): The type of model to train. Must be one of 'XGBoost', 'CatBoost', 'AdaBoost', or 'HistGradientBoosting'.

    Raises:
        ValueError: If the dataset does not contain the target column or if the model_type is not recognized.

    Returns:
        None
    """
    # Preprocess the data
    data = preprocessing.initial_preprocess(data)
    data = preprocessing.encode_categorical(data)
    data = preprocessing.correct_positive_skewness(data, ['carat', 'x', 'y'])

    # Define features and target
    features = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z']
    target = 'price'

    if target not in data.columns:
        raise ValueError(f"Dataset must contain the target column '{target}'")

    x = data[features]
    y = data[target]

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model based on the chosen type
    if model_type == 'XGBoost':
        model = train_xgboost(x_train, y_train)
    elif model_type == 'CatBoost':
        model = train_catboost(x_train, y_train)
    elif model_type == 'AdaBoost':
        model = train_adaboost(x_train, y_train)
    elif model_type == 'HistGradientBoosting':
        model = train_histgradientboosting(x_train, y_train)
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")

    # Save the trained model
    joblib.dump(model, 'models/user_model.pkl')
