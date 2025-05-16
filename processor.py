# processor.py
import json
import os
import time
from datetime import datetime, UTC
import pandas as pd
from confluent_kafka import Consumer, Producer
import configparser

# Load Kafka config
config = configparser.ConfigParser()
config.read('config.ini')

kafka_conf = {
    'bootstrap.servers': config['CONFLUENT_CLOUD']['bootstrap_servers'],
    'sasl.username': config['CONFLUENT_CLOUD']['kafka_api_key'],
    'sasl.password': config['CONFLUENT_CLOUD']['kafka_secret_key'],
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'group.id': 'processor-group',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(kafka_conf)
producer = Producer(kafka_conf)

RAW_TOPIC = 'raw-data'
PROCESSED_TOPIC = 'processed-data'

# File paths
OUTPUT_DIR = "generated_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROCESSED_FILENAME = f"{OUTPUT_DIR}/processed_features.csv"
INITIAL_FEATURES_FILE = f"{OUTPUT_DIR}/initial_features.csv"

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery to Kafka failed: {err}")
    else:
        print(f"Delivered to {msg.topic()} [{msg.partition()}]")

def send_to_kafka(topic, data):
    try:
        producer.produce(topic, value=json.dumps(data), callback=delivery_report)
        producer.poll(0)
    except Exception as e:
        print(f"Error sending to Kafka topic {topic}: {e}")

def process_message(message):
    try:
        data = json.loads(message.value().decode('utf-8'))

        total = data.get("resource_blocks_total")
        used = data.get("resource_blocks_used")

        if total and used:
            utilization = (used / total) * 100
            features = {
                "timestamp": data.get("timestamp"),
                "cell_id": data.get("cell_id"),
                "latency_ms": data.get("latency_ms"),
                "throughput_mbps": data.get("throughput_mbps"),
                "active_users": data.get("active_users"),
                "resource_block_utilization_percent": utilization
            }
            return features
    except Exception as e:
        print(f"Error processing message: {e}")
    return None

def main():
    print("Kafka processor started. Listening to 'raw-data' topic...")
    consumer.subscribe([RAW_TOPIC])
    processed_data = []

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            feature = process_message(msg)
            if feature:
                processed_data.append(feature)
                send_to_kafka(PROCESSED_TOPIC, feature)

                if len(processed_data) >= 50:
                    df = pd.DataFrame(processed_data)
                    if not os.path.exists(INITIAL_FEATURES_FILE):
                        df.to_csv(INITIAL_FEATURES_FILE, index=False)
                        print("Initial baseline data saved.")
                    df.to_csv(PROCESSED_FILENAME, mode='a', header=not os.path.exists(PROCESSED_FILENAME), index=False)
                    print(f"Saved {len(processed_data)} processed records to CSV.")
                    processed_data.clear()

    except KeyboardInterrupt:
        print("Stopping processor...")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
