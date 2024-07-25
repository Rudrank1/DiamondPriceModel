"""
Importing necessary packges
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import zscore

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
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
    Remove rows with invalid values (zeros).

    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with rows containing zero values removed.
    """
    return df[(df != 0).all(axis=1)]

def drop_duplicates(df):
    """
    Drop duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates()

def apply_bounds(df):
    """
    Apply boundaries to filter out out-of-bounds values.

    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with rows outside the defined bounds removed.
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
    Remove outliers based on Z-scores.

    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    z_scores = np.abs(zscore(df[numeric_columns]))
    return df[(z_scores < 3).all(axis=1)]

def initial_preprocess(df):
    """
    Apply initial preprocessing steps to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = remove_invalid_values(df)
    df = drop_duplicates(df)
    df = apply_bounds(df)
    df = remove_outliers(df)
    return df

def encode_categorical(df):
    """
    Encode categorical features as integers.

    Args:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with categorical features encoded.
    """
    label_encoder = preprocessing.LabelEncoder()
    df['cut'] = label_encoder.fit_transform(df['cut'])
    df['color'] = label_encoder.fit_transform(df['color'])
    df['clarity'] = label_encoder.fit_transform(df['clarity'])
    return df

def correct_positive_skewness(df, skewed_cols):
    """
    Apply log transformation to correct positive skewness in features.

    Args:
        df (pd.DataFrame): DataFrame to process.
        skewed_cols (list): List of column names to apply the log transformation.

    Returns:
        pd.DataFrame: DataFrame with corrected skewness.
    """
    df[skewed_cols] = np.log(df[skewed_cols])
    return df
