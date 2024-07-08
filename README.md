# Diamond Price Prediction

This Python script is designed to predict diamond prices using machine learning techniques. It utilizes the following libraries:

pandas for data manipulation
numpy for numerical computations
matplotlib for plotting
sklearn for preprocessing, model selection, and metrics
xgboost for gradient boosting
statsmodels for statistical computations

## Functions Provided:

load_data(file_path): Loads data from a CSV file into a Pandas DataFrame.

remove_invalid_values(df): Removes rows with zero values from the DataFrame.

drop_duplicates(df): Drops duplicate rows from the DataFrame.

apply_bounds(df): Filters the DataFrame based on domain-specific bounds for diamond characteristics.

remove_outliers(df): Uses the Z-score method to remove outliers from numeric columns.

initial_preprocess(df): Performs initial preprocessing steps including removing invalid values, dropping duplicates, applying bounds, and removing outliers.

encode_categorical(df): Encodes categorical features using LabelEncoder.

correct_positive_skewness(df, skewed_cols): Corrects positive skewness in specified columns using a logarithmic transformation.

calculate_vif(df): Calculates the Variance Inflation Factor (VIF) to detect multicollinearity and removes features with high VIF.

plot_residuals(y_test, y_pred): Plots a histogram of residuals between actual and predicted values.

train_and_evaluate_model(X_train, X_test, y_train, y_test): Trains an XGBoost regressor model, evaluates it using Root Mean Squared Error (RMSE), and plots residuals.

main(): Main function that orchestrates the entire pipeline from data loading to evaluation.


## How to Use:

Ensure you have Python installed along with the necessary libraries (pandas, numpy, matplotlib, sklearn, xgboost, statsmodels).

Place your dataset in CSV format named diamonds.csv in the same directory as the script. It should have the following columns:
1.   **Price:** US dollars (\$326--\$18,823)
2.  **Carat:** Weight of the diamond (0.2--5.01)
3.   **Cut:** Quality of the cut (Fair (worst), Good, Very Good, Premium, Ideal (best))
4.  **Color:** Diamond's colour, (J (worst), I, H, G, F, E, D (best)
5.  **Clarity:** Measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
6.  **x:** Length in mm. (0 to 10.74)
7.  **y:** Width in mm. (0 to 58.9)
8.  **z:** Depth in mm. (0 to 31.8)
9.  **Depth:** Total depth percentage, z divided by mean(x, y). (43 to 79)
10.  **Table:** Width of top of diamond relative to widest point. (43 to 95)

Run the script to obtain results.
