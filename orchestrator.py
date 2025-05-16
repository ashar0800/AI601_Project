# orchestrator.py
from prefect import flow, task
import subprocess
import time
import os

def wait_for_file(file_path, timeout=60, check_interval=5):
    """Wait for a file to exist with timeout."""
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for file: {file_path}")
        time.sleep(check_interval)
    # Additional wait to ensure file is completely written
    time.sleep(2)

@task
def start_producer():
    print("Starting producer.py")
    subprocess.Popen(["python", "producer.py"])
    time.sleep(45)  # Increased time to generate initial raw data

@task
def start_processor():
    print("Starting processor.py")
    subprocess.Popen(["python", "processor.py"])
    # Wait for processed features file
    wait_for_file("generated_files/processed_features.csv", timeout=120)
    print("Processed features file generated successfully")

@task
def start_trainer():
    print("Starting trainer.py")
    subprocess.Popen(["python", "trainer.py"])
    # Wait for model file
    wait_for_file("generated_files/resource_utilization_model.pkl", timeout=120)
    print("Model file generated successfully")

@task
def start_deployer():
    print("Starting deployer.py")
    subprocess.Popen(["python", "deployer.py"])
    time.sleep(15)  # Increased time for model deployment

@task
def start_drift_detector():
    print("Starting drift_detector.py")
    subprocess.Popen(["python", "drift_detector.py"])
    time.sleep(15)  # Increased time for drift detector initialization

@task
def start_alerter():
    print("Starting alerter.py")
    subprocess.Popen(["python", "alerter.py"])
    time.sleep(10)  # Increased time for alerter initialization

@task
def start_allocator():
    print("Starting allocator.py")
    subprocess.Popen(["python", "allocator.py"])
    time.sleep(10)  # Increased time for allocator initialization

@task
def start_ui():
    print("Starting ui.py")
    subprocess.Popen(["streamlit", "run", "ui.py"])
    time.sleep(10)  # Increased time for UI initialization

# Main Prefect flow
@flow(name="5G Resource Allocation Pipeline")
def main_flow():
    try:
        # Start in order, respecting dependencies and allowing headroom
        print("Starting pipeline components...")
        producer_future = start_producer()
        processor_future = start_processor(wait_for=[producer_future])
        trainer_future = start_trainer(wait_for=[processor_future])
        deployer_future = start_deployer(wait_for=[trainer_future, processor_future])
        drift_detector_future = start_drift_detector(wait_for=[deployer_future])
        alerter_future = start_alerter(wait_for=[drift_detector_future])
        allocator_future = start_allocator(wait_for=[deployer_future])
        ui_future = start_ui(wait_for=[processor_future, deployer_future, alerter_future, allocator_future])
        print("All components started successfully!")
    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main_flow()
