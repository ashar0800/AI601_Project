# drift_detector.py
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timezone
from scipy.stats import kstest
from sklearn.metrics import mean_squared_error
import json


# Define consistent file paths
OUTPUT_DIR = "generated_files"
INPUT_FEATURES_FILENAME = f"{OUTPUT_DIR}/processed_features.csv"
PREDICTIONS_FILENAME = f"{OUTPUT_DIR}/predictions.log"
BASELINE_DATA_FILENAME = f"{OUTPUT_DIR}/initial_features.csv"
DRIFT_LOG_FILENAME = f"{OUTPUT_DIR}/drift_detector_output.log"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration parameters
DRIFT_THRESHOLD_PSI = 0.1  # Threshold for Population Stability Index
DRIFT_CHECK_INTERVAL = 30  # Check for drift every 30 seconds
PERFORMANCE_EVALUATION_INTERVAL = 60  # Evaluate performance every 60 seconds
CONCEPT_DRIFT_MSE_THRESHOLD = 10.0  # Threshold for MSE increase


def write_to_log(message):
    """Writes a message to the drift detector log file."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat() + 'Z'
        log_message = f"{timestamp} - {message}"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(DRIFT_LOG_FILENAME), exist_ok=True)
        
        with open(DRIFT_LOG_FILENAME, "a") as f:
            f.write(log_message + "\n")
        print(log_message)
    except Exception as e:
        print(f"Error writing to log file: {e}")
        # Try to write to console as fallback
        print(f"Failed log message: {message}")


def calculate_psi(expected, actual, num_buckets=10):
    """Calculates the Population Stability Index (PSI)."""
    try:
        def calculate_bucket_counts(data, num_buckets):
            # Handle edge cases with data
            if len(data) == 0:
                write_to_log("Warning: Empty data array in PSI calculation")
                return np.zeros(num_buckets), np.linspace(0, 1, num_buckets + 1)
            
            # Remove any NaN or infinite values
            data = data[~np.isnan(data) & ~np.isinf(data)]
            if len(data) == 0:
                write_to_log("Warning: No valid data points after cleaning in PSI calculation")
                return np.zeros(num_buckets), np.linspace(0, 1, num_buckets + 1)
                
            hist, bin_edges = np.histogram(data, bins=num_buckets)
            # Add small epsilon to avoid division by zero
            counts = hist / (len(data) + 1e-10)
            return counts, bin_edges

        expected_counts, expected_bins = calculate_bucket_counts(expected, num_buckets)
        actual_counts, _ = calculate_bucket_counts(actual, num_buckets)  # Use same number of buckets

        psi_val = 0
        for i in range(num_buckets):
            # Add small epsilon to avoid log(0)
            if actual_counts[i] > 0 and expected_counts[i] > 0:
                psi_val += (actual_counts[i] - expected_counts[i]) * np.log((actual_counts[i] + 1e-10) / (expected_counts[i] + 1e-10))
        
        write_to_log(f"PSI calculation completed: {psi_val:.4f}")
        return psi_val
    except Exception as e:
        write_to_log(f"Error calculating PSI: {e}")
        return 0


def update_baseline_data(new_data_df):
    """Updates the baseline data with new data when drift is detected."""
    try:
        # Save new baseline data directly
        new_data_df.to_csv(BASELINE_DATA_FILENAME, index=False)
        write_to_log(f"Baseline data updated with new data distribution")
        return True
    except Exception as e:
        write_to_log(f"Error updating baseline data: {e}")
        return False


def detect_data_drift(baseline_df, current_df, feature_columns, threshold=DRIFT_THRESHOLD_PSI):
    """Detects data drift using PSI for specified features."""
    if baseline_df is None or current_df is None:
        write_to_log("Cannot detect drift: missing data")
        return False
        
    if baseline_df.empty or current_df.empty:
        write_to_log("Cannot detect drift: empty dataframes")
        return False
        
    drift_detected = {}
    for col in feature_columns:
        if col in baseline_df.columns and col in current_df.columns:
            # Skip if either column is empty
            if baseline_df[col].empty or current_df[col].empty:
                write_to_log(f"Skipping drift detection for {col}: empty data")
                continue
                
            # Remove any NaN or infinite values
            baseline_data = baseline_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            current_data = current_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(baseline_data) == 0 or len(current_data) == 0:
                write_to_log(f"Skipping drift detection for {col}: no valid data after cleaning")
                continue
                
            psi = calculate_psi(baseline_data, current_data)
            drift_detected[col] = psi > threshold
            
            if drift_detected[col]:
                write_to_log(f"Data drift detected for feature: {col} (PSI: {psi:.4f} > {threshold})")
            else:
                write_to_log(f"No drift detected for feature: {col} (PSI: {psi:.4f} <= {threshold})")
    
    # If drift is detected, update the baseline with current data
    if any(drift_detected.values()):
        write_to_log("Drift detected, updating baseline data...")
        # Use the last 50 rows of current data as new baseline
        new_baseline = current_df.tail(50)
        if update_baseline_data(new_baseline):
            write_to_log("Baseline data updated successfully")
        else:
            write_to_log("Failed to update baseline data")
                
    return any(drift_detected.values())


def load_predictions():
    """Loads predictions from the log file and converts timestamp to datetime."""
    predictions = []
    
    try:
        if not os.path.exists(PREDICTIONS_FILENAME):
            write_to_log(f"Predictions file not found: {PREDICTIONS_FILENAME}")
            return pd.DataFrame()
            
        with open(PREDICTIONS_FILENAME, "r") as f:
            for line in f:
                if "Predicted Utilization" in line:
                    try:
                        parts = line.split(", ")
                        timestamp_str = parts[0].split(": ")[1]
                        cell_id = parts[1].split(": ")[1]
                        prediction = float(parts[2].split(": ")[1].replace("%", ""))
                        predictions.append({"timestamp": timestamp_str, "cell_id": cell_id, "prediction": prediction})
                    except Exception as e:
                        write_to_log(f"Error parsing prediction line: {line.strip()}, error: {e}")
    except Exception as e:
        write_to_log(f"Error loading predictions: {e}")
        
    predictions_df = pd.DataFrame(predictions)
    if not predictions_df.empty and 'timestamp' in predictions_df.columns:
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
    return predictions_df


def simulate_ground_truth(df):
    """Simulates ground truth with a delay and some noise."""
    if df is None or df.empty or 'resource_block_utilization_percent' not in df.columns:
        return df
        
    try:
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Introduce a delay (shift by 5 rows)
        result_df['ground_truth'] = result_df['resource_block_utilization_percent'].shift(5)
        
        # Add noise
        result_df['ground_truth'] = result_df['ground_truth'] + np.random.normal(0, 0.02, len(result_df))
        
        # Remove rows with NaN values from the shift
        result_df.dropna(inplace=True)
        
        return result_df
    except Exception as e:
        write_to_log(f"Error simulating ground truth: {e}")
        return df


def detect_concept_drift(predictions_df, original_features_df, threshold=CONCEPT_DRIFT_MSE_THRESHOLD):
    """Detects concept drift by comparing predictions to (simulated) ground truth."""
    if predictions_df is None or original_features_df is None:
        write_to_log("Cannot detect concept drift: missing data")
        return False
        
    if predictions_df.empty or original_features_df.empty:
        write_to_log("Cannot detect concept drift: empty data")
        return False
        
    if 'timestamp' not in predictions_df.columns or 'timestamp' not in original_features_df.columns:
        write_to_log("Cannot detect concept drift: missing timestamp column")
        return False
        
    if 'prediction' not in predictions_df.columns or 'ground_truth' not in original_features_df.columns:
        write_to_log("Cannot detect concept drift: missing prediction or ground truth columns")
        return False

    try:
        # Ensure timestamps are in datetime format
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        original_features_df['timestamp'] = pd.to_datetime(original_features_df['timestamp'])
        
        # Merge predictions and ground truth based on timestamp
        merged_df = pd.merge_asof(
            predictions_df.sort_values('timestamp'), 
            original_features_df[['timestamp', 'ground_truth']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Drop rows with missing values
        merged_df.dropna(subset=['prediction', 'ground_truth'], inplace=True)
        
        if merged_df.empty:
            write_to_log("No matching data points for concept drift detection")
            return False
            
        # Calculate mean squared error
        mse = mean_squared_error(merged_df['ground_truth'], merged_df['prediction'])
        write_to_log(f"Current Model MSE: {mse:.4f}")
        
        # Detect concept drift if MSE exceeds threshold
        if mse > threshold:
            write_to_log(f"Concept drift detected! MSE ({mse:.4f}) exceeds threshold ({threshold})")
            # Update baseline with current data when concept drift is detected
            new_baseline = original_features_df.tail(50)
            update_baseline_data(new_baseline)
            return True
            
        return False
    except Exception as e:
        write_to_log(f"Error detecting concept drift: {e}")
        return False


def load_processed_features(filename=INPUT_FEATURES_FILENAME):
    """Loads the latest processed features and converts timestamp to datetime."""
    try:
        if not os.path.exists(filename):
            write_to_log(f"Processed features file not found: {filename}")
            return None
            
        df = pd.read_csv(filename)
        
        if df.empty:
            write_to_log(f"Processed features file is empty: {filename}")
            return None
            
        if 'timestamp' in df.columns:
            # Parse timestamps with flexible format
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            
        return df
    except Exception as e:
        write_to_log(f"Error loading processed features: {e}")
        return None


def get_dynamic_drift_threshold(default=DRIFT_THRESHOLD_PSI):
    config_path = os.path.join("data_drift_threshold_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                return float(config.get("drift_threshold_psi", default))
    except Exception as e:
        write_to_log(f"Error reading drift threshold config: {e}")
    return default


def get_dynamic_concept_drift_threshold(default=CONCEPT_DRIFT_MSE_THRESHOLD):
    config_path = os.path.join("concept_drift_threshold_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                return float(config.get("concept_drift_mse_threshold", default))
    except Exception as e:
        write_to_log(f"Error reading concept drift threshold config: {e}")
    return default


def main():
    """Main function to monitor for data and concept drift."""
    print("Drift detector started.")
    write_to_log("Drift detector service started")
    
    # Load or wait for initial data as baseline
    baseline_df = None
    while baseline_df is None:
        # First, try to load from the baseline file if it exists
        if os.path.exists(BASELINE_DATA_FILENAME):
            baseline_df = pd.read_csv(BASELINE_DATA_FILENAME)
            write_to_log(f"Baseline data loaded from {BASELINE_DATA_FILENAME}")
        else:
            # Otherwise, try to load from the processed features file
            initial_features_df = load_processed_features(INPUT_FEATURES_FILENAME)
            if initial_features_df is not None and not initial_features_df.empty:
                baseline_df = initial_features_df.copy()
                
                # Save this as the baseline for future reference
                try:
                    baseline_df.to_csv(BASELINE_DATA_FILENAME, index=False)
                    write_to_log(f"Baseline data saved to {BASELINE_DATA_FILENAME}")
                except Exception as e:
                    write_to_log(f"Error saving baseline data: {e}")
            else:
                write_to_log("Waiting for initial data to establish baseline...")
                time.sleep(10)  # Wait before retrying

    # Start the monitoring loop
    last_performance_check = time.time()
    
    while True:
        try:
            # Check for data drift
            current_features_df = load_processed_features(INPUT_FEATURES_FILENAME)
            if current_features_df is not None and not current_features_df.empty:
                drift_threshold = get_dynamic_drift_threshold()
                detect_data_drift(
                    baseline_df, 
                    current_features_df, 
                    ['latency_ms', 'throughput_mbps', 'active_users'],
                    threshold=drift_threshold
                )
            
            # Periodically check for concept drift
            current_time = time.time()
            if current_time - last_performance_check >= PERFORMANCE_EVALUATION_INTERVAL:
                predictions_df = load_predictions()
                processed_features_for_gt = load_processed_features(INPUT_FEATURES_FILENAME)
                
                if processed_features_for_gt is not None and not processed_features_for_gt.empty:
                    ground_truth_df = simulate_ground_truth(processed_features_for_gt)
                    concept_drift_threshold = get_dynamic_concept_drift_threshold()
                    detect_concept_drift(predictions_df, ground_truth_df, threshold=concept_drift_threshold)
                    
                last_performance_check = current_time
                
            # Wait before next check
            time.sleep(DRIFT_CHECK_INTERVAL)
            
        except Exception as e:
            write_to_log(f"Unexpected error in drift detector: {e}")
            time.sleep(DRIFT_CHECK_INTERVAL)  # Continue despite errors


if __name__ == "__main__":
    main()
