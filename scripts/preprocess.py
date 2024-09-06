import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    """
    Preprocess the data by removing null values and normalizing the 'rooms' and 'sqft' columns.
    
    Args:
    - input_path (str): The file path for the input CSV data.
    - output_path (str): The file path where the processed CSV data will be saved.
    """
    # Load dataset
    data = pd.read_csv(input_path)

    # Remove null values
    data = data.dropna()

    # Normalize the data (using StandardScaler to standardize features)
    scaler = StandardScaler()
    data[['rooms', 'sqft']] = scaler.fit_transform(data[['rooms', 'sqft']])

    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Preprocessed data saved at {output_path}")

if __name__ == "__main__":
    # Define input and output paths
    input_path = "/opt/ml/processing/input/rental_1000.csv"
    output_path = "/opt/ml/processing/output/processed_data.csv"

    # Call the function to preprocess the data
    preprocess_data(input_path, output_path)
