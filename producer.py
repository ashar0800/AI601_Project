# producer.py
import json
import random
import time
from datetime import datetime, UTC
from confluent_kafka import Producer
import configparser
import os

# Load Kafka config
config = configparser.ConfigParser()
config.read('config.ini')

kafka_conf = {
    'bootstrap.servers': config['CONFLUENT_CLOUD']['bootstrap_servers'],
    'sasl.username': config['CONFLUENT_CLOUD']['kafka_api_key'],
    'sasl.password': config['CONFLUENT_CLOUD']['kafka_secret_key'],
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN'
}

producer = Producer(kafka_conf)
RAW_TOPIC = 'raw-data'

# Mapping of cell IDs to location keys
CELL_KEYS = {
    "cell-1": "DHA Phase 1",
    "cell-2": "DHA Phase 2",
    "cell-3": "DHA Phase 3",
    "cell-4": "DHA Phase 4",
    "cell-5": "DHA Phase 5"
}

# Simulated phase counter for drift progression
phase_counter = {"value": 0}

def generate_data():
    # Simulate drift by increasing phase value over time
    phase = phase_counter["value"]
    phase_counter["value"] += 1

    cell_id = random.choice(list(CELL_KEYS.keys()))
    timestamp = datetime.now(UTC).isoformat()
    resource_blocks_total = 100

    if phase < 500:
        resource_blocks_used = random.randint(10, 90)
        latency_ms = round(random.uniform(5, 50), 2)
        throughput_mbps = round(random.uniform(10, 500), 2)
        active_users = random.randint(50, 500)
    elif 500 <= phase < 1000:
        resource_blocks_used = random.randint(30, 95)
        latency_ms = round(random.uniform(20, 80), 2)
        throughput_mbps = round(random.uniform(5, 200), 2)
        active_users = random.randint(100, 600)
    elif 1000 <= phase < 1500:
        resource_blocks_used = random.randint(20, 80)
        active_users = random.randint(150, 700)
        base_latency = round(random.uniform(10, 40), 2)
        latency_ms = base_latency + (active_users / 10)
        throughput_mbps = round(random.uniform(20, 400), 2)
    else:
        resource_blocks_used = random.randint(5, 60)
        latency_ms = round(random.uniform(15, 65), 2)
        throughput_mbps = round(random.uniform(150, 650), 2)
        active_users = random.randint(20, 300)

    data = {
        "timestamp": timestamp,
        "cell_id": cell_id,
        "resource_blocks_total": resource_blocks_total,
        "resource_blocks_used": resource_blocks_used,
        "latency_ms": latency_ms,
        "throughput_mbps": throughput_mbps,
        "active_users": active_users
    }

    return data, CELL_KEYS[cell_id]

def write_to_local_file(data, key, filename="generated_files/network_data.log"):
    """
    Writes network data to a local log file.
    Args:
        data: Dictionary containing network data
        filename: Path to the log file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert data to JSON string and write with newline
        with open(filename, "a") as f:
            f.write("Cell ID: " + key + ": " + json.dumps(data) + "\n")
            
        # Print confirmation (optional, for debugging)
        # print(f"Data written to {filename}")
    except Exception as e:
        print(f"Error writing to file {filename}: {e}")

def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}] Key: {msg.key()}")

def main():
    print("Producing messages to Kafka topic...")
    while True:
        data, key = generate_data() 
        try:
            # Write to local file
            write_to_local_file(data,key)
            
            # Send to Kafka
            producer.produce(
                topic=RAW_TOPIC,
                key=key,  # Send key as plain string, not bytes
                value=json.dumps(data),
                callback=delivery_report
            )
            producer.poll(0)
        except Exception as e:
            print(f"Error producing message: {e}")
        time.sleep(1)

if __name__ == "__main__":
    main()
