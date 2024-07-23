import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
from scipy.stats import zscore
import joblib

def load_data(file_path):
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
    return df[(df != 0).all(axis=1)]

def drop_duplicates(df):
    return df.drop_duplicates()

def apply_bounds(df):
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
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    z_scores = np.abs(zscore(df[numeric_columns]))
    return df[(z_scores < 3).all(axis=1)]

def initial_preprocess(df):
    df = remove_invalid_values(df)
    df = drop_duplicates(df)
    df = apply_bounds(df)
    df = remove_outliers(df)
    return df

def encode_categorical(df):
    label_encoder = preprocessing.LabelEncoder()
    df['cut'] = label_encoder.fit_transform(df['cut'])
    df['color'] = label_encoder.fit_transform(df['color'])
    df['clarity'] = label_encoder.fit_transform(df['clarity'])
    return df

def correct_positive_skewness(df, skewed_cols):
    df[skewed_cols] = np.log(df[skewed_cols])
    return df

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()

def train_and_evaluate_model(x_train, x_test, y_train, y_test):
    param_dist = {
        'num_leaves': [31, 41, 51, 61],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300, 400, 500],
        'min_child_samples': [20, 30, 40, 50]
    }
    
    lgbm = lgb.LGBMRegressor(random_state=1, verbose=0)
    
    random_search = model_selection.RandomizedSearchCV(
        estimator=lgbm,
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
    
    joblib.dump(best_model, 'lightgbm_model.pkl')
    
    print('RMSE: ' + str(rmse))
    
    plot_residuals(y_test, y_pred)

def main():
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
