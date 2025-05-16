# trainer.py
import pandas as pd
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime


# Define consistent file paths
OUTPUT_DIR = "generated_files"
MODEL_FILENAME = f"{OUTPUT_DIR}/resource_utilization_model.pkl"
FEATURE_FILENAME = f"{OUTPUT_DIR}/processed_features.csv"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the feature and target columns
FEATURE_COLUMNS = ["latency_ms", "throughput_mbps", "active_users"]
TARGET_COLUMN = "resource_block_utilization_percent"


def load_data(filename=FEATURE_FILENAME):
    """Loads data from a CSV file."""
    try:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None
            
        df = pd.read_csv(filename)
        if df.empty:
            print(f"No data in file: {filename}")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None


def train_model(df):
    """Trains a linear regression model."""
    if df is None or df.empty:
        print("Error: No data to train the model.")
        return None

    # Ensure all required columns are present
    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in data: {missing_cols}")
        return None
        
    try:
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model trained. Mean Squared Error on test set: {mse:.2f}")

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None


def save_model(model, filename=MODEL_FILENAME):
    """Saves the trained model to a pickle file."""
    if model is None:
        print("Error: Cannot save None model.")
        return False
        
    try:
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"Trained model saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving model to {filename}: {e}")
        return False


def load_model(filename=MODEL_FILENAME):
    """Loads a trained model from a pickle file."""
    try:
        if not os.path.exists(filename):
            print(f"Model file not found: {filename}")
            return None
            
        with open(filename, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model from {filename}: {e}")
        return None


def retrain_model(filename=FEATURE_FILENAME):
    """Loads new data and retrains the model."""
    print("Initiating model retraining...")
    new_df = load_data(filename)
    if new_df is not None and not new_df.empty:
        retrained_model = train_model(new_df)
        if retrained_model:
            success = save_model(retrained_model)
            if success:
                print("Model retraining complete.")
                return True
            else:
                print("Failed to save retrained model.")
                return False
        else:
            print("Retraining failed.")
            return False
    else:
        print("No new data available for retraining.")
        return False


def main():
    """Main function to train the initial model and perform periodic retraining."""
    print("Model trainer started.")
    
    # Training initial model
    print("Initial model training...")
    initial_data = load_data()
    if initial_data is not None:
        initial_model = train_model(initial_data)
        if initial_model:
            save_model(initial_model)

    # Periodic retraining
    while True:
        time.sleep(60)  # Wait for 60 seconds before retraining
        print("Performing periodic retraining...")
        retrain_model()


if __name__ == "__main__":
    main()
