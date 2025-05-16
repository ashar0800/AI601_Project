# deployer.py
import pandas as pd
import os
import time
import pickle


# Define consistent file paths
OUTPUT_DIR = "generated_files"
MODEL_FILENAME = f"{OUTPUT_DIR}/resource_utilization_model.pkl"
INPUT_FEATURES_FILENAME = f"{OUTPUT_DIR}/processed_features.csv"
PREDICTIONS_FILENAME = f"{OUTPUT_DIR}/predictions.log"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define feature columns (same as in trainer.py)
FEATURE_COLUMNS = ["latency_ms", "throughput_mbps", "active_users"]


def load_model(model_filename=MODEL_FILENAME):
    """Loads a trained model from a pickle file."""
    try:
        if not os.path.exists(model_filename):
            print(f"Model file not found: {model_filename}")
            return None
            
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_resource_utilization(model, features):
    """Predicts resource utilization using the loaded model."""
    try:
        if model is None:
            print("Error: Model is None, cannot make prediction.")
            return None
            
        # Check if all features are present
        missing_cols = [col for col in FEATURE_COLUMNS if col not in features.columns]
        if missing_cols:
            print(f"Error: Missing features for prediction: {missing_cols}")
            return None
            
        prediction = model.predict(features[FEATURE_COLUMNS])[0]
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def read_and_predict(model_filename=MODEL_FILENAME, input_filename=INPUT_FEATURES_FILENAME, output_filename=PREDICTIONS_FILENAME):
    """Reads processed features from CSV, performs prediction, and logs the results."""
    model = load_model(model_filename)
    if model is None:
        print("Cannot proceed without a valid model.")
        time.sleep(10)  # Wait before retrying
        return

    try:
        # Keep track of the last processed row to avoid reprocessing
        last_processed_row = 0
        
        # Check if input file exists
        if not os.path.exists(input_filename):
            print(f"Input file {input_filename} not found. Waiting for data...")
            time.sleep(5)
            return
            
        # Read the data
        df = pd.read_csv(input_filename)
        
        if len(df) > last_processed_row:
            new_data = df.iloc[last_processed_row:]
            for index, row in new_data.iterrows():
                try:
                    # Create a DataFrame from the row for prediction
                    features = row[FEATURE_COLUMNS].to_frame().T
                    
                    if not features.empty and all(col in features.columns for col in FEATURE_COLUMNS):
                        prediction = predict_resource_utilization(model, features)
                        
                        if prediction is not None:
                            timestamp = row['timestamp']
                            cell_id = row['cell_id']
                            log_message = f"Timestamp: {timestamp}, Cell ID: {cell_id}, Predicted Utilization: {prediction:.2f}%"
                            print(log_message)
                            
                            # Write prediction to log file
                            with open(output_filename, "a") as outfile:
                                outfile.write(log_message + "\n")
                    else:
                        print(f"Skipping incomplete or irrelevant data at index {index}")
                except Exception as e:
                    print(f"An error occurred during prediction for row {index}: {e}")
            
            # Update the last processed row
            last_processed_row = len(df)
    except pd.errors.EmptyDataError:
        print(f"Warning: {input_filename} is empty.")
    except Exception as e:
        print(f"An error occurred while processing: {e}")


def main():
    """Main function to load the model and start real-time prediction."""
    print("Model deployer started.")
    
    while True:
        read_and_predict()
        time.sleep(5)  # Check for new data every 5 seconds


if __name__ == "__main__":
    main()
