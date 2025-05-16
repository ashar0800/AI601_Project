# alerter.py
import time
import os
from datetime import datetime, timezone
import json


# Configuration: File paths and operational parameters
DRIFT_DETECTOR_LOG_FILENAME = "generated_files/drift_detector_output.log"
ALERTS_LOG_FILENAME = "generated_files/alerts.log"
CHECK_INTERVAL_SECONDS = 10  # How often to check for new drift messages

# Default allocation thresholds for predicted utilization
DEFAULT_HIGH_UTILIZATION_THRESHOLD = 85.0  # Percentage: if predicted utilization is above this, consider increasing resources
DEFAULT_LOW_UTILIZATION_THRESHOLD = 30.0   # Percentage: if predicted utilization is below this, consider decreasing resources

# To keep track of the last processed line number in the drift detector log
last_processed_line_number = 0


def create_alert_message(drift_log_line):
    """
    Creates a formatted alert message from a drift detector log line.
    Returns the alert message string or None if the line doesn't indicate a notable drift.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    alert_prefix = f"{timestamp} - ALERT: "

    try:
        # Standardize the drift message for the alert log
        if "Data drift detected for feature:" in drift_log_line:
            # Example: "Data drift detected for feature: latency_ms (PSI: 0.1234 > 0.1)"
            try:
                # Extract relevant parts
                parts = drift_log_line.split("Data drift detected for feature:")[1].split("(PSI:")
                feature = parts[0].strip()
                comparison_str = parts[1].strip().rstrip(")")
                
                # Split the comparison string into value and threshold
                if ">" in comparison_str:
                    psi_str, threshold_str = comparison_str.split(">")
                    psi_str = psi_str.strip()
                    threshold_str = threshold_str.strip()
                    
                    # Try to convert to float for formatting, but keep original string if it fails
                    try:
                        psi_float = float(psi_str)
                        threshold_float = float(threshold_str)
                        return f"{alert_prefix}Data drift detected for feature '{feature}' (PSI: {psi_float:.4f} > {threshold_float:.4f})."
                    except ValueError:
                        return f"{alert_prefix}Data drift detected for feature '{feature}' (PSI: {psi_str} > {threshold_str})."
                else:
                    return f"{alert_prefix}Data drift detected for feature '{feature}' (PSI: {comparison_str})."
            except IndexError:
                # Fallback if parsing fails
                return f"{alert_prefix}Data drift detected. Details: {drift_log_line.strip()}"

        elif "Concept drift detected" in drift_log_line:
            # Example: "Concept drift detected! MSE (12.3456) exceeds threshold (10.0)"
            try:
                # Extract MSE value
                mse_start = drift_log_line.find("MSE (") + 5
                mse_end = drift_log_line.find(")", mse_start)
                mse_str = drift_log_line[mse_start:mse_end].strip()
                
                # Extract threshold value
                threshold_start = drift_log_line.find("threshold (") + 11
                threshold_end = drift_log_line.find(")", threshold_start)
                threshold_str = drift_log_line[threshold_start:threshold_end].strip()
                
                # Try to convert to float for formatting, but keep original string if it fails
                try:
                    mse_float = float(mse_str)
                    threshold_float = float(threshold_str)
                    return f"{alert_prefix}Concept drift detected (MSE: {mse_float:.4f} > {threshold_float:.4f})."
                except ValueError:
                    return f"{alert_prefix}Concept drift detected (MSE: {mse_str} > {threshold_str})."
            except IndexError:
                # Fallback if parsing fails
                return f"{alert_prefix}Concept drift detected. Details: {drift_log_line.strip()}"

        elif "Warning:" in drift_log_line:
            # Handle warning messages
            warning_msg = drift_log_line.split("Warning:")[1].strip()
            return f"{alert_prefix}WARNING: {warning_msg}"

        elif "Error:" in drift_log_line:
            # Handle error messages
            error_msg = drift_log_line.split("Error:")[1].strip()
            return f"{alert_prefix}ERROR: {error_msg}"

        # Add more conditions here if drift_detector.py logs other types of critical alerts

    except Exception as e:
        print(f"Error creating alert message: {e}")
        return f"{alert_prefix}Error parsing drift message: {drift_log_line.strip()}"

    return None  # Not a line we want to generate an alert for


def write_to_alerts_log(message):
    """
    Appends the given alert message to the ALERTS_LOG_FILENAME.
    """
    try:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(ALERTS_LOG_FILENAME), exist_ok=True)
        with open(ALERTS_LOG_FILENAME, "a") as f:
            f.write(message + "\n")
        print(f"Alert logged: {message}")
    except Exception as e:
        print(f"Error writing to alerts log '{ALERTS_LOG_FILENAME}': {e}")


def check_for_new_drift_events():
    """
    Checks the drift detector log for new drift events since the last check
    and logs alerts if any are found.
    """
    global last_processed_line_number
    new_alerts_generated = False
    try:
        # Check if the drift detector log file exists and is not empty
        if not os.path.exists(DRIFT_DETECTOR_LOG_FILENAME):
            print(f"Drift detector log file {DRIFT_DETECTOR_LOG_FILENAME} not found. It will be created when drift_detector.py detects drift")
            return False

        if os.path.getsize(DRIFT_DETECTOR_LOG_FILENAME) == 0:
            print(f"Drift detector log file {DRIFT_DETECTOR_LOG_FILENAME} is empty.")
            return False

        with open(DRIFT_DETECTOR_LOG_FILENAME, "r") as f:
            lines = f.readlines()
        current_line_count = len(lines)

        if current_line_count > last_processed_line_number:
            # Process new lines
            new_lines = lines[last_processed_line_number:]
            print(f"Found {len(new_lines)} new lines in {DRIFT_DETECTOR_LOG_FILENAME}.")
            
            for line in new_lines:
                stripped_line = line.strip()
                if stripped_line:  # Ensure line is not empty
                    try:
                        alert_message = create_alert_message(stripped_line)
                        if alert_message:
                            write_to_alerts_log(alert_message)
                            new_alerts_generated = True
                    except Exception as e:
                        print(f"Error processing line: {stripped_line}, error: {e}")
                        # Try to write a basic alert for the error
                        write_to_alerts_log(f"{datetime.now(timezone.utc).isoformat()} - ALERT: Error processing drift message: {e}")
            
            last_processed_line_number = current_line_count
            if new_alerts_generated:
                print("New alerts have been generated and logged.")
    except FileNotFoundError:
        print(f"Drift detector log file {DRIFT_DETECTOR_LOG_FILENAME} not found. It will be created when drift_detector.py detects drift")
        return False
    except Exception as e:
        print(f"Error reading drift detector log '{DRIFT_DETECTOR_LOG_FILENAME}': {e}")
        return False
    return new_alerts_generated


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


def check_utilization(cell_id, predicted_utilization, prediction_timestamp_str):
    """
    Checks if the predicted utilization is outside acceptable thresholds.
    Returns a log message string for the alerts log if an alert is needed, otherwise None.
    """
    # Get current thresholds from config
    high_threshold, low_threshold = get_dynamic_utilization_thresholds()
    
    if predicted_utilization > high_threshold:
        alert_message = (
            f"{datetime.now(timezone.utc).isoformat()} - Cell_ID: {cell_id}, "
            f"Triggering_Prediction_Timestamp: {prediction_timestamp_str}, "
            f"Alert_Type: HIGH_UTILIZATION, "
            f"Predicted_Utilization: {predicted_utilization:.2f}%, "
            f"Threshold: {high_threshold:.2f}%"
        )
        return alert_message
    elif predicted_utilization < low_threshold:
        alert_message = (
            f"{datetime.now(timezone.utc).isoformat()} - Cell_ID: {cell_id}, "
            f"Triggering_Prediction_Timestamp: {prediction_timestamp_str}, "
            f"Alert_Type: LOW_UTILIZATION, "
            f"Predicted_Utilization: {predicted_utilization:.2f}%, "
            f"Threshold: {low_threshold:.2f}%"
        )
        return alert_message
    return None


def main():
    """
    Main function for the alerter.
    Continuously monitors the drift detector log and generates alerts.
    """
    print("Alerter service started. Monitoring for drift detection events...")

    # Ensure log directories exist
    try:
        os.makedirs(os.path.dirname(DRIFT_DETECTOR_LOG_FILENAME), exist_ok=True)
        os.makedirs(os.path.dirname(ALERTS_LOG_FILENAME), exist_ok=True)
        print("Log directories created successfully.")
    except OSError as e:
        print(f"Error creating log directories: {e}. Please ensure the program has write permissions.")
        return  # Exit if directories cannot be created

    while True:
        try:
            check_for_new_drift_events()
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("Alerter service shutting down due to KeyboardInterrupt.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the alerter main loop: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS * 2)


if __name__ == "__main__":
    main()
