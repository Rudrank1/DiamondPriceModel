"""
Importing necessary packges
"""
from sklearn import model_selection
import preprocessing
from modeling import train_and_evaluate_models

def main():
    """
    Main function to execute data preprocessing and model training.

    Loads the dataset, applies preprocessing, trains models with hyperparameter tuning.
    """
    file_path = 'diamonds.csv'
    diamond_data = preprocessing.load_data(file_path)

    diamond_data = preprocessing.initial_preprocess(diamond_data)
    diamond_data = preprocessing.encode_categorical(diamond_data)
    diamond_data = preprocessing.correct_positive_skewness(diamond_data, ['carat', 'x', 'y'])

    features = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z']
    x = diamond_data[features]
    y = diamond_data['price']

    x1, x2, y1, y2 = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)
    train_and_evaluate_models(x1, x2, y1, y2)

if __name__ == "__main__":
    main()
