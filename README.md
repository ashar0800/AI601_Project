---

# 📡 5G Dynamic Resource Allocation Simulation

This project simulates dynamic resource allocation in 5G cellular networks using real-time data streams, machine learning, drift detection, and responsive visualization. It integrates a full Kafka-based pipeline, predictive modeling, drift monitoring, and an interactive dashboard to provide actionable insights into network health and utilization.

---

## 🛠️ Components Overview

| File / Module       | Purpose                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| `producer.py`       | Simulates and streams raw network data to Kafka                             |
| `processor.py`      | Consumes raw data, computes features, sends processed data to Kafka and CSV |
| `trainer.py`        | Trains a linear regression model on historical network features             |
| `deployer.py`       | Loads the model and performs real-time predictions                          |
| `drift_detector.py` | Monitors for data and concept drift using PSI and MSE                       |
| `alerter.py`        | Parses drift logs and raises alerts                                         |
| `allocator.py`      | Decides allocation actions based on predicted utilization                   |
| `ui.py`             | Streamlit dashboard visualizing network status, alerts, predictions         |
| `orchestrator.py`   | Launches the full pipeline using Prefect orchestration                      |

---

## 📁 Data & Logs

| File / Log                       | Schema / Fields                                                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `raw-data` (Kafka)               | `timestamp`, `cell_id`, `resource_blocks_total`, `resource_blocks_used`, `latency_ms`, `throughput_mbps`, `active_users` |
| `processed_features.csv`         | `timestamp`, `cell_id`, `latency_ms`, `throughput_mbps`, `active_users`, `resource_block_utilization_percent`            |
| `predictions.log`                | `Timestamp`, `Cell ID`, `Predicted Utilization`                                                                          |
| `network_data.log`               | `Cell ID`, JSON (same as raw network data)                                                                               |
| `drift_detector_output.log`      | `timestamp`, `feature`, `psi`, `mse`, `threshold`                                                                        |
| `allocation.log`                 | `decision_timestamp`, `cell_id`, `prediction_timestamp`, `action`, `predicted_utilization`, `reason`                     |
| `alerts.log`                     | `timestamp`, `cell_id`, `prediction_timestamp`, `alert_type`, `predicted_utilization`, `threshold`, etc.                 |
| `resource_utilization_model.pkl` | Pickled scikit-learn model                                                                                               |

---

## 🚀 Quick Start

1. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Kafka Configuration**

   * Add your Kafka credentials to `config.ini` under `[CONFLUENT_CLOUD]`.

3. **Start Server**

   ```bash
   prefect server start
   ```
   
4. **Run Orchestrator (Recommended)**

   ```bash
   python orchestrator.py
   ```

5. **OR Start Services Manually**

   ```bash
   python producer.py
   python processor.py
   python trainer.py
   python deployer.py
   python drift_detector.py
   python alerter.py
   python allocator.py
   streamlit run ui.py
   ```

---

## 📊 Dashboard

The Streamlit dashboard displays several kinds of metrics:

* Current network metrics (utilization, latency, users)
* Latest predictions & allocation actions
* Alerts for data/concept drift
* Historical trends and log viewers

---

## 🧠 ML Model

* **Algorithm**: Linear Regression
* **Target**: `resource_block_utilization_percent`
* **Features**: `latency_ms`, `throughput_mbps`, `active_users`
* **Retrains Periodically** using latest data.

---

## 🧪 Drift Detection

* **Data Drift**: Population Stability Index (PSI) > 0.1
* **Concept Drift**: MSE of predictions vs simulated ground truth > 10.0

---

## 📌 Requirements

* Python 3.8+
* `confluent_kafka`, `pandas`, `scikit-learn`, `streamlit`, `prefect`, `numpy`, etc.

---

## 🗃️ Folder Structure

```
generated_files/
│
├── processed_features.csv
├── predictions.log
├── allocation.log
├── alerts.log
├── drift_detector_output.log
├── network_data.log
├── initial_features.csv
└── resource_utilization_model.pkl
```

---

## 📄 License

MIT License — use freely, cite if helpful.

---

Let me know if you'd like a `requirements.txt`, Dockerfile, or deployment guide!
