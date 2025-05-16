# ui.py
import streamlit as st
import pandas as pd
import time
import json
import re
from datetime import datetime, timezone
import os
import subprocess

# Configuration Files
PREDICTIONS_LOG_FILENAME = "generated_files/predictions.log"
ALERTS_LOG_FILENAME = "generated_files/alerts.log"
ALLOCATION_LOG_FILENAME = "generated_files/allocation.log"
NETWORK_DATA_LOG_FILENAME = "generated_files/network_data.log"
DRIFT_DETECTOR_LOG_FILENAME = "generated_files/drift_detector_output.log"
PROCESSED_FEATURES_FILENAME = "generated_files/processed_features.csv"

# Default parameter values
DEFAULT_DRIFT_THRESHOLD_PSI = 0.1
DEFAULT_CONCEPT_DRIFT_MSE_THRESHOLD = 10.0
DEFAULT_HIGH_UTILIZATION_THRESHOLD = 85.0
DEFAULT_LOW_UTILIZATION_THRESHOLD = 30.0

def load_latest_data(filename, parser_function=lambda x: x):
    try:
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            lines = f.readlines()
            if lines:
                return parser_function(lines[-1].strip())
    except Exception as e:
        st.error(f"Error loading data from {filename}: {e}")
        return None
    return None

def parse_network_data(line):
    """Parse a line from network_data.log into a dictionary."""
    try:
        # Extract the JSON part after "Cell ID: {key}: "
        json_str = line.split(": ", 2)[-1]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        st.error(f"Error parsing network data: {e}")
        return None

def get_latest_prediction():
    line = load_latest_data(PREDICTIONS_LOG_FILENAME)
    if line:
        match = re.search(r"Cell ID: (.*?), Predicted Utilization: (.*?)\%", line)
        if match:
            return match.group(1).strip(), float(match.group(2).strip())
    return None, None

def get_latest_alert():
    alert = load_latest_data(ALERTS_LOG_FILENAME)
    if alert:
        try:
            # Split timestamp and message
            timestamp, message = alert.split(" - ", 1)
            # Parse the alert message
            if "Data drift detected" in message:
                match = re.search(r"Data drift detected for feature '(.+?)' \(PSI: (.+?) > (.+?)\)\.", message)
                if match:
                    feature = match.group(1)
                    psi_str = match.group(2)
                    threshold_str = match.group(3)
                    try:
                        psi = float(psi_str)
                        threshold = float(threshold_str)
                        return {
                            "timestamp": timestamp,
                            "type": "Data Drift",
                            "feature": feature,
                            "psi": psi,
                            "threshold": threshold,
                            "message": message
                        }
                    except ValueError:
                        return {
                            "timestamp": timestamp,
                            "message": message
                        }
                else:
                    # Try to parse the PSI value directly if the full pattern doesn't match
                    try:
                        feature = message.split("'")[1]
                        psi_part = message.split("PSI: ")[1]
                        psi_value = psi_part.split(" > ")[0].strip()
                        return {
                            "timestamp": timestamp,
                            "type": "Data Drift",
                            "feature": feature,
                            "psi": float(psi_value),
                            "message": message
                        }
                    except Exception:
                        return {
                            "timestamp": timestamp,
                            "message": message
                        }
            return {
                "timestamp": timestamp,
                "message": message
            }
        except Exception as e:
            st.error(f"Error parsing alert: {e}")
            return None
    return None

def get_latest_allocation():
    allocation = load_latest_data(ALLOCATION_LOG_FILENAME)
    if allocation:
        try:
            # Split timestamp and message
            timestamp, message = allocation.split(" - ", 1)
            # Parse the allocation message
            parts = message.split(", ")
            allocation_data = {}
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    allocation_data[key.strip()] = value.strip()
            
            return {
                "timestamp": timestamp,
                "cell_id": allocation_data.get("Cell_ID", "N/A"),
                "action": allocation_data.get("Action", "N/A"),
                "predicted_utilization": allocation_data.get("Predicted_Utilization", "N/A"),
                "reason": allocation_data.get("Reason", "N/A"),
                "message": message
            }
        except Exception as e:
            st.error(f"Error parsing allocation: {e}")
            return None
    return None

def read_log_file(filename, max_lines=100):
    try:
        if not os.path.exists(filename):
            return ["Log file not found."]
        with open(filename, "r") as f:
            lines = f.readlines()
            if not lines:
                return ["No data in log file."]
            
            # Get the last max_lines
            recent_lines = lines[-max_lines:]
            
            # Format each line
            formatted_lines = []
            for line in recent_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Try to parse timestamp if present
                if " - " in line:
                    timestamp, message = line.split(" - ", 1)
                    try:
                        # Handle both formats: with and without timezone
                        if timestamp.endswith('Z'):
                            # Remove Z and add +00:00 for consistent parsing
                            timestamp = timestamp[:-1] + '+00:00'
                        elif '+' not in timestamp and 'Z' not in timestamp:
                            # Add UTC timezone if none present
                            timestamp = timestamp + '+00:00'
                        
                        dt = datetime.fromisoformat(timestamp)
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Special formatting for different log types
                        if "Cell ID:" in message and "Predicted Utilization:" in message:
                            # Format prediction logs
                            cell_id = message.split("Cell ID:")[1].split(",")[0].strip()
                            utilization = message.split("Predicted Utilization:")[1].split("%")[0].strip()
                            formatted_lines.append(
                                f"🕒 {formatted_time}\n"
                                f"📊 Prediction Update\n"
                                f"📱 Cell: {cell_id}\n"
                                f"📈 Predicted Utilization: {utilization}%\n"
                            )
                        elif "Cell ID:" in message and ":" in message:
                            # Format network data logs
                            try:
                                data = json.loads(message.split(": ", 1)[1])
                                formatted_lines.append(
                                    f"🕒 {formatted_time}\n"
                                    f"🌐 Network Status\n"
                                    f"📱 Cell: {data.get('cell_id', 'N/A')}\n"
                                    f"📊 Utilization: {data.get('resource_blocks_used', 0)}/{data.get('resource_blocks_total', 0)} blocks\n"
                                    f"⏱️ Latency: {data.get('latency_ms', 'N/A')} ms\n"
                                    f"📡 Throughput: {data.get('throughput_mbps', 'N/A')} Mbps\n"
                                    f"👥 Active Users: {data.get('active_users', 'N/A')}\n"
                                )
                            except:
                                formatted_lines.append(f"🕒 {formatted_time}\n{message}\n")
                        else:
                            formatted_lines.append(f"🕒 {formatted_time}\n{message}\n")
                    except Exception as e:
                        # If parsing fails, just show the original line
                        formatted_lines.append(f"{line}\n")
                else:
                    formatted_lines.append(f"{line}\n")
            
            return formatted_lines
    except Exception as e:
        return [f"Error reading log file: {e}"]

def is_script_running(script_name):
    try:
        if os.name == 'nt':
            tasks = subprocess.check_output(['tasklist'], creationflags=subprocess.CREATE_NO_WINDOW, text=True)
            return f"python*{script_name}.py" in tasks
        else:
            processes = subprocess.check_output(['pgrep', '-f', f"{script_name}.py"], text=True)
            return script_name in processes
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        st.error(f"Error checking script status for {script_name}: {e}")
        return False

def ensure_directories_exist():
    try:
        os.makedirs("generated_files", exist_ok=True)
        return True
    except Exception as e:
        st.error(f"Failed to create necessary directories: {e}")
        return False

def load_processed_features(max_rows=50):
    try:
        if not os.path.exists(PROCESSED_FEATURES_FILENAME):
            return None
        df = pd.read_csv(PROCESSED_FEATURES_FILENAME)
        if df.empty:
            return None
        return df.tail(max_rows)
    except Exception as e:
        st.error(f"Error loading processed features: {e}")
        return None

def write_utilization_thresholds(high_threshold, low_threshold):
    """Write utilization thresholds to config file."""
    config = {
        "high_utilization_threshold": high_threshold,
        "low_utilization_threshold": low_threshold
    }
    with open("utilization_threshold_config.json", "w") as f:
        json.dump(config, f)

def main():
    ensure_directories_exist()

    st.title("5G Dynamic Resource Allocation Simulation Dashboard")

    # Sidebar controls
    st.sidebar.header("Simulation Controls")

    drift_threshold_psi = st.sidebar.slider(
        "Drift Threshold (PSI)", 0.01, 0.5, DEFAULT_DRIFT_THRESHOLD_PSI, 0.01
    )
    # Write drift threshold to config file
    drift_config_path = "data_drift_threshold_config.json"
    try:
        with open(drift_config_path, "w") as f:
            json.dump({"drift_threshold_psi": drift_threshold_psi}, f)
    except Exception as e:
        st.error(f"Failed to write drift threshold config: {e}")

    concept_drift_mse_threshold = st.sidebar.number_input(
        "Concept Drift Threshold (MSE)", 10.0, 600.0, DEFAULT_CONCEPT_DRIFT_MSE_THRESHOLD, 5.0
    )
    # Write concept drift threshold to config file
    concept_drift_config_path = "concept_drift_threshold_config.json"
    try:
        with open(concept_drift_config_path, "w") as f:
            json.dump({"concept_drift_mse_threshold": concept_drift_mse_threshold}, f)
    except Exception as e:
        st.error(f"Failed to write concept drift threshold config: {e}")

    # Add utilization threshold sliders
    st.sidebar.markdown("### Utilization Thresholds")
    high_threshold = st.sidebar.slider(
        "High Utilization Threshold (%)",
        min_value=50.0,
        max_value=95.0,
        value=85.0,
        step=1.0,
        help="Threshold above which high utilization alerts will be triggered"
    )
    low_threshold = st.sidebar.slider(
        "Low Utilization Threshold (%)",
        min_value=5.0,
        max_value=49.0,
        value=30.0,
        step=1.0,
        help="Threshold below which low utilization alerts will be triggered"
    )
    
    # Ensure low threshold is always less than high threshold
    if low_threshold >= high_threshold:
        st.sidebar.error("Low threshold must be less than high threshold")
        return
    
    # Write thresholds to config file
    write_utilization_thresholds(high_threshold, low_threshold)

    # Current metrics
    st.subheader("Current Network Status")
    col1, col2, col3 = st.columns(3)

    latest_network_data = load_latest_data(NETWORK_DATA_LOG_FILENAME, parse_network_data)
    if latest_network_data:
        cell_id = latest_network_data.get("cell_id", "N/A")
        resource_blocks_used = latest_network_data.get("resource_blocks_used", 0)
        resource_blocks_total = latest_network_data.get("resource_blocks_total", 1)

        utilization = (
            round(resource_blocks_used / resource_blocks_total * 100, 2)
            if resource_blocks_total > 0 else "N/A"
        )
        latency = latest_network_data.get("latency_ms", "N/A")
        throughput = latest_network_data.get("throughput_mbps", "N/A")

        col1.metric("Cell ID", cell_id)
        col2.metric("Utilization (%)", utilization)
        col3.metric("Latency (ms)", latency)

        col1, col2, col3 = st.columns(3)
        col1.metric("Throughput (Mbps)", throughput)
        col2.metric("Active Users", latest_network_data.get("active_users", "N/A"))

    prediction_cell, predicted_utilization = get_latest_prediction()
    if prediction_cell:
        st.metric(f"Predicted Utilization (%)", f"{predicted_utilization:.2f}")

    latest_alert = get_latest_alert()
    if latest_alert:
        if isinstance(latest_alert, dict):
            st.markdown("""
                <style>
                .alert-header {
                    color:rgb(255, 255, 255);
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .alert-content {
                    color:rgb(255, 255, 255);
                }
                .alert-metric {
                    color:rgb(255, 255, 255);
                    font-weight: bold;
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="alert-header">⚠️ Latest Alert</div>', unsafe_allow_html=True)
            
            if latest_alert.get("type") == "Data Drift":
                st.markdown(f"""
                    <div class="alert-content">
                        <p>Data drift detected for feature: <span class="alert-metric">{latest_alert['feature']}</span></p>
                        <p>PSI Score: <span class="alert-metric">{latest_alert['psi']:.4f}</span></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-content">{latest_alert["message"]}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Latest Alert: {latest_alert}")

    latest_allocation = get_latest_allocation()
    if latest_allocation:
        st.markdown("""
            <style>
            .allocation-box {
                background-color:rgb(255, 255, 255);
                border: 1px solidrgb(0, 0, 0);
                border-radius: 0px;
                padding: 0px;
                margin: 0px 0;
            }
            .allocation-header {
                color:white;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .allocation-content {
                color: white;
            }
            .allocation-metric {
                color:rgb(255, 255, 255);
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="allocation-header">📊 Latest Resource Allocation', unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="allocation-content">
                <p>Cell: <span class="allocation-metric">{latest_allocation['cell_id']}</span></p>
                <p>Action: <span class="allocation-metric">{latest_allocation['action']}</span></p>
                <p>Predicted Utilization: <span class="allocation-metric">{latest_allocation['predicted_utilization']}</span></p>
                <p>Reason: {latest_allocation['reason']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Log viewer
    st.subheader("Log Data")
    log_file_selection = st.selectbox(
        "Select Log File",
        ["Predictions", "Alerts", "Allocation", "Network Data", "Drift Detector"]
    )
    
    # Add slider for number of logs to display
    # num_logs = st.slider("Number of logs to display",min_value=1,max_value=50,value=5,help="Select how many recent logs to display")
    
    log_files = {
        "Predictions": PREDICTIONS_LOG_FILENAME,
        "Alerts": ALERTS_LOG_FILENAME,
        "Allocation": ALLOCATION_LOG_FILENAME,
        "Network Data": NETWORK_DATA_LOG_FILENAME,
        "Drift Detector": DRIFT_DETECTOR_LOG_FILENAME,
    }
    
    # Add custom styling for the log viewer
    st.markdown("""
        <style>
        .log-container {
            background-color: #1E1E1E;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Consolas', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .log-entry {
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 4px;
            background-color: #2D2D2D;
            border-left: 4px solid #4CAF50;
        }
        .log-timestamp {
            color: #9CDCFE;
            font-size: 0.9em;
            margin-bottom: 8px;
        }
        .log-message {
            margin-top: 8px;
            color: #D4D4D4;
        }
        .network-status {
            color: #4EC9B0;
        }
        .network-metric {
            color: #9CDCFE;
            margin-left: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    log_lines = read_log_file(log_files[log_file_selection], max_lines=5)
    
    # Create a container with custom styling
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    for line in log_lines:
        if "ALERT:" in line:
            st.markdown(f'<div class="log-entry" style="border-left-color: #FF6B6B;">⚠️ {line}</div>', unsafe_allow_html=True)
        elif "INFO:" in line:
            st.markdown(f'<div class="log-entry" style="border-left-color: #4CAF50;">ℹ️ {line}</div>', unsafe_allow_html=True)
        elif "ERROR:" in line:
            st.markdown(f'<div class="log-entry" style="border-left-color: #FF6B6B;">❌ {line}</div>', unsafe_allow_html=True)
        elif "Network Status" in line:
            # Split the formatted network status into lines and apply special styling
            lines = line.split('\n')
            st.markdown('<div class="log-entry" style="border-left-color: #4EC9B0;">', unsafe_allow_html=True)
            for i, l in enumerate(lines):
                if i == 0:  # Timestamp
                    st.markdown(f'<div class="log-timestamp">{l}</div>', unsafe_allow_html=True)
                elif i == 1:  # Network Status header
                    st.markdown(f'<div class="network-status">{l}</div>', unsafe_allow_html=True)
                else:  # Metrics
                    st.markdown(f'<div class="network-metric">{l}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif "Prediction Update" in line:
            st.markdown(f'<div class="log-entry" style="border-left-color: #9CDCFE;">{line}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="log-entry">{line}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Historical charts
    st.subheader("Historical Trends")
    df_features = load_processed_features(max_rows=50)

    tab1, tab2 = st.tabs(["Raw Network Data", "Processed Features"])

    with tab1:
        try:
            historical_data = []
            with open(NETWORK_DATA_LOG_FILENAME, "r") as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    try:
                        data = parse_network_data(line.strip())
                        if data:  # Only append if parsing was successful
                            historical_data.append(data)
                    except Exception as e:
                        st.error(f"Error processing line: {e}")
                        continue
                        
            if historical_data:
                df_network_history = pd.DataFrame(historical_data)
                if (
                    "timestamp" in df_network_history.columns and
                    "resource_blocks_used" in df_network_history.columns and
                    "resource_blocks_total" in df_network_history.columns
                ):
                    df_network_history["utilization_percent"] = (
                        df_network_history["resource_blocks_used"]
                        / df_network_history["resource_blocks_total"]
                        * 100
                    )
                    df_network_history["timestamp"] = pd.to_datetime(df_network_history["timestamp"])
                    st.line_chart(
                        df_network_history.set_index("timestamp")[[
                            "utilization_percent", "latency_ms", "throughput_mbps"
                        ]]
                    )
        except Exception as e:
            st.error(f"Error processing historical data: {e}")

    with tab2:
        if df_features is not None and not df_features.empty:
            if "timestamp" in df_features.columns:
                df_features["timestamp"] = pd.to_datetime(df_features["timestamp"])
                df_features = df_features.set_index("timestamp")

            if "resource_block_utilization_percent" in df_features.columns:
                st.line_chart(df_features["resource_block_utilization_percent"])

            if all(col in df_features.columns for col in ["latency_ms", "throughput_mbps", "active_users"]):
                st.line_chart(df_features[["latency_ms", "throughput_mbps", "active_users"]])
        else:
            st.warning("No processed feature data available for visualization.")

    st.checkbox("Auto-refresh (5s)", value=True, key="auto_refresh")

    if st.session_state.get("auto_refresh", True):
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
