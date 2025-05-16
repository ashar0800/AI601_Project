# allocator.py
import time
import re
from datetime import datetime, timezone
import os
import json


# Configuration: File paths and operational parameters
PREDICTIONS_LOG_FILENAME = "generated_files/predictions.log"
ALLOCATION_LOG_FILENAME = "generated_files/allocation.log"
CHECK_INTERVAL_SECONDS = 10  # How often to check for new predictions (in seconds)

# Default allocation thresholds for predicted utilization
DEFAULT_HIGH_UTILIZATION_THRESHOLD = 85.0  # Percentage: if predicted utilization is above this, consider increasing resources
DEFAULT_LOW_UTILIZATION_THRESHOLD = 30.0   # Percentage: if predicted utilization is below this, consider decreasing resources


# Global variable to store the timestamp of the last processed prediction
# This helps in avoiding processing the same prediction multiple times.
last_processed_prediction_timestamp = None


def parse_prediction_log_line(line):
    """
    Parses a single line from the predictions.log file.
    Expected format: "Timestamp: YYYY-MM-DDTHH:MM:SS.ffffffZ, Cell ID: Cell-X, Predicted Utilization: XX.YY%"
    Returns: (timestamp_str, cell_id, predicted_utilization) or (None, None, None) if parsing fails.
    """
    # Regex to capture the required fields from the log line
    match = re.search(
        r"Timestamp: (.*?), Cell ID: (.*?), Predicted Utilization: (.*?)%",
        line
    )
    if match:
        timestamp_str = match.group(1).strip()
        cell_id = match.group(2).strip()
        predicted_utilization_str = match.group(3).strip()
        try:
            predicted_utilization = float(predicted_utilization_str)
            # Validate timestamp format (optional, but good practice)
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return timestamp_str, cell_id, predicted_utilization
        except ValueError as e:
            print(f"Error parsing values from prediction line: '{line.strip()}'. Error: {e}")
            return None, None, None
    else:
        # print(f"Line did not match expected prediction format: '{line.strip()}'") # Can be noisy
        pass
    return None, None, None


def get_latest_prediction():
    """
    Reads the latest prediction from PREDICTIONS_LOG_FILENAME.
    Returns the latest prediction (timestamp, cell_id, utilization) if it's new,
    otherwise (None, None, None).
    """
    global last_processed_prediction_timestamp
    try:
        # Check if the prediction file exists and is not empty
        if not os.path.exists(PREDICTIONS_LOG_FILENAME) or os.path.getsize(PREDICTIONS_LOG_FILENAME) == 0:
            # This is a normal condition if deployer.py hasn't run yet or produced output
            return None, None, None

        with open(PREDICTIONS_LOG_FILENAME, "r") as f:
            lines = f.readlines()
            if not lines:
                return None, None, None  # File might have been emptied
           
            latest_line = lines[-1].strip()  # Get the last line

        # Parse the latest line
        timestamp_str, cell_id, predicted_utilization = parse_prediction_log_line(latest_line)

        if timestamp_str:
            # Check if this prediction has already been processed
            if timestamp_str != last_processed_prediction_timestamp:
                last_processed_prediction_timestamp = timestamp_str  # Update to the current timestamp
                return timestamp_str, cell_id, predicted_utilization
            else:
                # Prediction with this timestamp already processed, do nothing
                return None, None, None
        else:
            # Parsing failed for the latest line
            return None, None, None

    except FileNotFoundError:
        # print(f"Prediction file {PREDICTIONS_LOG_FILENAME} not found. Waiting for it to be created.")
        return None, None, None
    except Exception as e:
        print(f"Error reading predictions log '{PREDICTIONS_LOG_FILENAME}': {e}")
        return None, None, None


def get_dynamic_utilization_thresholds():
    """Reads utilization thresholds from config file."""
    config_path = "utilization_threshold_config.json"
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                return (
                    float(config.get("high_utilization_threshold", DEFAULT_HIGH_UTILIZATION_THRESHOLD)),
                    float(config.get("low_utilization_threshold", DEFAULT_LOW_UTILIZATION_THRESHOLD))
                )
    except Exception as e:
        print(f"Error reading utilization threshold config: {e}")
    return DEFAULT_HIGH_UTILIZATION_THRESHOLD, DEFAULT_LOW_UTILIZATION_THRESHOLD


def decide_allocation_action(cell_id, predicted_utilization, prediction_timestamp_str):
    """
    Decides on an allocation action based on the predicted utilization.
    Returns a log message string for the allocation log.
    """
    # Get current thresholds from config
    high_threshold, low_threshold = get_dynamic_utilization_thresholds()
    
    action = "MAINTAIN_CURRENT_RESOURCES"
    reason = f"Predicted utilization {predicted_utilization:.2f}% is within normal operating thresholds ({low_threshold:.2f}% - {high_threshold:.2f}%)."

    if predicted_utilization > high_threshold:
        action = "SUGGEST_INCREASE_RESOURCES"
        reason = f"Predicted utilization {predicted_utilization:.2f}% exceeded high threshold ({high_threshold:.2f}%)."
    elif predicted_utilization < low_threshold:
        action = "SUGGEST_DECREASE_RESOURCES"
        reason = f"Predicted utilization {predicted_utilization:.2f}% is below low threshold ({low_threshold:.2f}%)."

    # Timestamp for when this allocation decision is being made
    current_decision_timestamp = datetime.now(timezone.utc).isoformat()

    allocation_log_message = (
        f"{current_decision_timestamp} - Cell_ID: {cell_id}, "
        f"Triggering_Prediction_Timestamp: {prediction_timestamp_str}, Action: {action}, "
        f"Predicted_Utilization: {predicted_utilization:.2f}%, Reason: {reason}"
    )
    return allocation_log_message


def write_to_allocation_log(message):
    """
    Appends the given message to the ALLOCATION_LOG_FILENAME.
    """
    try:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(ALLOCATION_LOG_FILENAME), exist_ok=True)
        with open(ALLOCATION_LOG_FILENAME, "a") as f:
            f.write(message + "\n")
        print(f"Allocation logged: {message}")
    except Exception as e:
        print(f"Error writing to allocation log '{ALLOCATION_LOG_FILENAME}': {e}")


def main():
    """
    Main function for the allocator.
    Continuously monitors predictions and logs allocation decisions.
    """
    print("Allocator service started. Monitoring for new predictions...")
   
    # Ensure log directories exist
    try:
        os.makedirs(os.path.dirname(PREDICTIONS_LOG_FILENAME), exist_ok=True)
        os.makedirs(os.path.dirname(ALLOCATION_LOG_FILENAME), exist_ok=True)
        print("Log directories created successfully.")
    except OSError as e:
        print(f"Warning: Could not create directory for log files: {e}. Ensure 'generated_files' directory exists and is writable.")

    while True:
        try:
            # Get the latest prediction if it's new
            prediction_timestamp, cell_id, predicted_utilization = get_latest_prediction()

            if prediction_timestamp and cell_id is not None and predicted_utilization is not None:
                print(f"New prediction received - Timestamp: {prediction_timestamp}, Cell: {cell_id}, Predicted Utilization: {predicted_utilization:.2f}%")
               
                # Decide on the allocation action based on the new prediction
                allocation_message = decide_allocation_action(cell_id, predicted_utilization, prediction_timestamp)
               
                if allocation_message:
                    write_to_allocation_log(allocation_message)
            # else:
                # No new prediction, or an error occurred while fetching, or prediction already processed.
                # print("No new predictions to process at this moment.")
           
            # Wait for the defined interval before checking again
            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("Allocator service shutting down due to KeyboardInterrupt.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the allocator main loop: {e}")
            # In case of an unexpected error, wait before retrying to prevent rapid error loops
            time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
