import time
import joblib
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import matplotlib.pyplot as plt

def plot_residuals(y_tests, y_preds, model_names):
    """
    Plot residuals for each model and save to the 'Data' folder.
    """

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, (y_test, y_pred, model_name) in enumerate(zip(y_tests, y_preds, model_names)):
        residuals = y_test - y_pred
        axes[i].scatter(y_pred, residuals, alpha=0.5)
        axes[i].hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r')
        axes[i].set_xlabel('Predicted values')
        axes[i].set_ylabel('Residuals')
        axes[i].set_title(f'Residuals for {model_name}')
    
    plt.tight_layout()
    plt.savefig('Data/residual_plots.png')  # Save residual plots as a PNG file
    plt.close()  # Close the plot to free memory

def train_lightgbm(x_train, y_train):
    param_grid = {
        'num_leaves': [31, 41, 51, 61],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300, 400, 500],
        'min_child_samples': [20, 30, 40, 50]
    }
    model = lgb.LGBMRegressor(random_state=1, verbose=-1)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=1
    )
    return search.fit(x_train, y_train)

def train_xgboost(x_train, y_train):
    param_grid = {
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
    model = xgb.XGBRegressor(random_state=1, verbosity=0)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=1
    )
    return search.fit(x_train, y_train)

def train_catboost(x_train, y_train):
    param_grid = {
        'iterations': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
        'depth': [3, 4, 5, 6, 7, 8],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128]
    }
    model = cat.CatBoostRegressor(
        random_state=1,
        logging_level='Silent'
    )
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=1
    )
    return search.fit(x_train, y_train)

def train_adaboost(x_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }
    model = AdaBoostRegressor(random_state=1)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    return search.fit(x_train, y_train)

def train_histgradientboosting(x_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'max_iter': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 7, 9],
        'min_samples_leaf': [1, 2, 4]
    }
    model = HistGradientBoostingRegressor(random_state=1)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=1
    )
    return search.fit(x_train, y_train)

def train_and_evaluate_models(x_train, x_test, y_train, y_test):
    model_functions = {
        'LightGBM': train_lightgbm,
        'XGBoost': train_xgboost,
        'CatBoost': train_catboost,
        'AdaBoost': train_adaboost,
        'HistGradientBoosting': train_histgradientboosting
    }
    
    best_models = {}
    y_tests = []
    y_preds = []
    model_names = []
    rmse_results = []

    for name, func in model_functions.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        search = func(x_train, y_train)
        best_model = search.best_estimator_
        
        best_models[name] = best_model
        # Save the model with compression in the 'models' folder
        joblib.dump(best_model, f'models/{name}_model.pkl', compress=3)
        
        y_pred = best_model.predict(x_test)
        y_preds.append(y_pred)
        y_tests.append(y_test)
        model_names.append(name)
        
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse_results.append((name, rmse))
        
        elapsed_time = time.time() - start_time
        print(f'Training time for {name}: {elapsed_time:.2f} seconds')
    
    plot_residuals(y_tests, y_preds, model_names)
    
    # Save RMSE results to a text file
    with open('Data/rmse_results.txt', 'w') as file:
        for name, rmse in rmse_results:
            file.write(f'RMSE for {name}: {rmse}\n')