import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import sklearn.ensemble as ensemble
import xgboost as xgb
import sklearn.metrics as metrics
from math import sqrt
from scipy.stats import zscore
import statsmodels.stats.outliers_influence as outliers_influence

def load_data(file_path):
    diamond = pd.read_csv(file_path, index_col=0, header=0)
    return diamond

def remove_invalid_values(df):
    df = df[(df != 0).all(axis=1)]
    return df

def drop_duplicates(df):
    df = df.drop_duplicates()
    return df

def apply_bounds(df):
    price_condition = (df['price'] >= 326) & (df['price'] <= 18823)
    carat_condition = (df['carat'] >= 0.2) & (df['carat'] <= 5.01)
    cut_condition = df['cut'].isin(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color_condition = df['color'].isin(['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity_condition = df['clarity'].isin(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    x_condition = df['x'] <= 10.74
    y_condition = df['y'] <= 58.9
    z_condition = df['z'] <= 31.8
    depth_condition = (df['depth'] >= 43) & (df['depth'] <= 79)
    table_condition = (df['table'] >= 43) & (df['table'] <= 95)

    df = df[carat_condition & cut_condition & color_condition & clarity_condition &
            depth_condition & table_condition & price_condition & x_condition &
            y_condition & z_condition]
    return df

def remove_outliers(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    z_scores = np.abs(zscore(df[numeric_columns]))
    df_no_outliers = df[(z_scores < 3).all(axis=1)]
    return df_no_outliers

def initial_preprocess(df):
    df = remove_invalid_values(df)
    df = drop_duplicates(df)
    df = apply_bounds(df)
    df = remove_outliers(df)
    return df

def encode_categorical(df):
    label_encoder = preprocessing.LabelEncoder()
    cut_order = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_order = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_order = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    df['cut_mapped'] = df['cut'].map(cut_order)
    df['color_mapped'] = df['color'].map(color_order)
    df['clarity_mapped'] = df['clarity'].map(clarity_order)
    df['cut'] = label_encoder.fit_transform(df['cut_mapped'])
    df['color'] = label_encoder.fit_transform(df['color_mapped'])
    df['clarity'] = label_encoder.fit_transform(df['clarity_mapped'])
    df.drop(['cut_mapped', 'color_mapped', 'clarity_mapped'], axis=1, inplace=True)
    return df

def correct_positive_skewness(df, skewed_cols):
    df[skewed_cols] = np.log(df[skewed_cols])
    return df

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [outliers_influence.variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    high_vif_cols = vif_data[vif_data['VIF'] > 400]['feature'].tolist()
    df.drop(high_vif_cols, axis=1, inplace=True)
    return df

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    xgboost = xgb.XGBRegressor(n_estimators=300, learning_rate=0.04, 
                                 min_child_weight=4, subsample=0.8, 
                                 colsample_bytree=0.8, random_state=1)
    xgboost.fit(X_train, y_train)
    y_pred = xgboost.predict(X_test)
    plot_residuals(y_test, y_pred)
    rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))
    return rmse

def main():
    diamond_data = load_data('diamonds.csv')
    diamond_data = initial_preprocess(diamond_data)

    diamond_data = encode_categorical(diamond_data)
    diamond_data = correct_positive_skewness(diamond_data, ['carat', 'table', 'x', 'y'])
    diamond_data = calculate_vif(diamond_data)

    X = diamond_data.drop('price', axis=1)
    y = diamond_data['price']
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.2, random_state=1)
    
    rmse = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    print("RMSE: " + str(rmse))

    plt.show()

if __name__ == "__main__":
    main()