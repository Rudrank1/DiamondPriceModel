"""
Importing all necessary libraries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, metrics
import xgboost as xgb
from scipy.stats import zscore
import joblib

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.

    Raises:
    FileNotFoundError: If the file path does not exist.
    pd.errors.EmptyDataError: If no data is found in the CSV file.
    pd.errors.ParserError: If there is an error parsing the CSV file.
    """
    try:
        diamond = pd.read_csv(file_path, index_col=0, header=0)
    except FileNotFoundError as exc:
        print("The file path does not exist.")
        raise exc
    except pd.errors.EmptyDataError:
        print("No data found in the CSV file.")
        raise
    except pd.errors.ParserError:
        print("Error parsing the CSV file.")
        raise
    return diamond

def remove_invalid_values(df):
    """
    Remove rows with zero values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with rows containing zero values removed.
    """
    return df[(df != 0).all(axis=1)]

def drop_duplicates(df):
    """
    Remove duplicate rows.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates()

def apply_bounds(df):
    """
    Apply bounds to filter the DataFrame based on domain knowledge.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The filtered DataFrame based on predefined bounds and conditions.
    """
    conditions = {
        'price': (326, 18823),
        'carat': (0.2, 5.01),
        'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
        'color': ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
        'x': (None, 10.74),
        'y': (None, 58.9),
        'z': (None, 31.8),
        'depth': (43, 79),
        'table': (43, 95)
    }

    df = df[(df['price'].between(*conditions['price'])) &
            (df['carat'].between(*conditions['carat'])) &
            (df['cut'].isin(conditions['cut'])) &
            (df['color'].isin(conditions['color'])) &
            (df['clarity'].isin(conditions['clarity'])) &
            (df['x'] <= conditions['x'][1]) &
            (df['y'] <= conditions['y'][1]) &
            (df['z'] <= conditions['z'][1]) &
            (df['depth'].between(*conditions['depth'])) &
            (df['table'].between(*conditions['table']))]
    return df

def remove_outliers(df):
    """
    Remove outliers using Z-score method.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    """
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    z_scores = np.abs(zscore(df[numeric_columns]))
    return df[(z_scores < 3).all(axis=1)]

def initial_preprocess(df):
    """
    Perform initial preprocessing: remove invalid values, outliers, drop duplicates, apply bounds.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    df = remove_invalid_values(df)
    df = drop_duplicates(df)
    df = apply_bounds(df)
    df = remove_outliers(df)
    return df

def encode_categorical(df):
    """
    Encode categorical features using LabelEncoder.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with encoded categorical features.
    """
    label_encoder = preprocessing.LabelEncoder()
    df['cut'] = label_encoder.fit_transform(df['cut'])
    df['color'] = label_encoder.fit_transform(df['color'])
    df['clarity'] = label_encoder.fit_transform(df['clarity'])
    return df

def correct_positive_skewness(df, skewed_cols):
    """
    Correct positive skewness in specified columns using logarithmic transformation.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    skewed_cols (list): List of column names to apply the transformation.

    Returns:
    pd.DataFrame: The DataFrame with corrected skewness.
    """
    df[skewed_cols] = np.log(df[skewed_cols])
    return df

def plot_residuals(y_test, y_pred):
    """
    Plot histogram of residuals.

    Parameters:
    y_test (pd.Series): Actual values.
    y_pred (pd.Series): Predicted values.
    """
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()

def train_and_evaluate_model(x_train, x_test, y_train, y_test):
    """
    Train and evaluate the XGBoost model with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    x_train (np.ndarray): Training features.
    x_test (np.ndarray): Testing features.
    y_train (pd.Series): Training target.
    y_test (pd.Series): Testing target.

    Returns:
    float: Root Mean Squared Error (RMSE) of the model.
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 2, 3, 4, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [1, 0.1, 0.01, 0]
    }
    
    xgboost = xgb.XGBRegressor(random_state=1)
    
    random_search = model_selection.RandomizedSearchCV(
        estimator=xgboost,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=1
    )
    random_search.fit(x_train, y_train)
    best_model = random_search.best_estimator_
    
    y_pred = best_model.predict(x_test)
    
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    joblib.dump(best_model, 'xgboost_model.pkl')
    
    print('RMSE: ' + str(rmse))
    
    plot_residuals(y_test, y_pred)

def main():
    """
    Main function to load data, preprocess, train, and evaluate the model.
    """
    file_path = 'diamonds.csv'
    diamond_data = load_data(file_path)

    diamond_data = initial_preprocess(diamond_data)
    diamond_data = encode_categorical(diamond_data)
    diamond_data = correct_positive_skewness(diamond_data, ['carat' ,'x', 'y'])

    features = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z']
    x = diamond_data[features]
    y = diamond_data['price']

    x1, x2, y1, y2 = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)
    train_and_evaluate_model(x1, x2, y1, y2)

if __name__ == "__main__":
    main()
