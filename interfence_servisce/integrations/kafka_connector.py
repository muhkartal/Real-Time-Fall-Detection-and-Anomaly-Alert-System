#!/usr/bin/env python3
"""
Kafka integration for EdgeVision-Guard.

This module provides a Kafka producer and consumer for integrating 
EdgeVision-Guard with enterprise event buses.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

import confluent_kafka
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME", "")
KAFKA_PASSWORD = os.getenv("KAFKA_PASSWORD", "")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "edgevision-guard")
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() in ("true", "1", "yes")

# Topics
KAFKA_ANOMALY_TOPIC = os.getenv("KAFKA_ANOMALY_TOPIC", "edgevision.anomalies")
KAFKA_TELEMETRY_TOPIC = os.getenv("KAFKA_TELEMETRY_TOPIC", "edgevision.telemetry")
KAFKA_COMMAND_TOPIC = os.getenv("KAFKA_COMMAND_TOPIC", "edgevision.commands")


class KafkaProducer:
    """Kafka producer for sending messages to Kafka topics."""
    
    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        security_protocol: str = KAFKA_SECURITY_PROTOCOL,
        sasl_mechanism: str = KAFKA_SASL_MECHANISM,
        username: str = KAFKA_USERNAME,
        password: str = KAFKA_PASSWORD,
    ):
        """
        Initialize the Kafka producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            security_protocol: Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)
            sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512)
            username: SASL username
            password: SASL password
        """
        # Check if Kafka is enabled
        if not KAFKA_ENABLED:
            logger.warning("Kafka integration is disabled. Set KAFKA_ENABLED=true to enable.")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Configure Kafka producer
        config = {
            "bootstrap.servers": bootstrap_servers,
            "client.id": f"edgevision-producer-{os.getpid()}",
            "security.protocol": security_protocol,
        }
        
        # Add SASL configuration if using SASL
        if "SASL" in security_protocol:
            config.update({
                "sasl.mechanism": sasl_mechanism,
                "sasl.username": username,
                "sasl.password": password,
            })
        
        # Create Kafka producer
        try:
            self.producer = Producer(config)
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.enabled = False
            raise
    
    def delivery_report(self, err: Any, msg: Any) -> None:
        """
        Delivery report callback for Kafka producer.
        
        Args:
            err: Error object (None if successful)
            msg: Message object
        """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
    
    def send_message(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ) -> bool:
        """
        Send a message to a Kafka topic.
        
        Args:
            topic: Kafka topic
            value: Message value (will be JSON serialized)
            key: Message key
            headers: Message headers
        
        Returns:
            True if message was sent, False otherwise
        """
        if not self.enabled:
            logger.warning("Kafka integration is disabled. Message not sent.")
            return False
        
        try:
            # Serialize message value to JSON
            value_bytes = json.dumps(value).encode("utf-8")
            
            # Serialize key if provided
            key_bytes = key.encode("utf-8") if key else None
            
            # Send message
            self.producer.produce(
                topic=topic,
                key=key_bytes,
                value=value_bytes,
                headers=headers,
                callback=self.delivery_report,
            )
            
            # Flush to ensure message is sent
            self.producer.poll(0)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            return False
    
    def send_anomaly(
        self,
        anomaly_data: Dict[str, Any],
        device_id: Optional[str] = None,
    ) -> bool:
        """
        Send an anomaly detection to the anomaly topic.
        
        Args:
            anomaly_data: Anomaly data
            device_id: Device ID (used as message key)
        
        Returns:
            True if message was sent, False otherwise
        """
        # Add timestamp if not present
        if "timestamp" not in anomaly_data:
            anomaly_data["timestamp"] = time.time()
        
        # Add metadata
        anomaly_data["metadata"] = {
            "source": "edgevision-guard",
            "version": "1.0.0",
            "device_id": device_id or os.getenv("DEVICE_ID", "unknown"),
        }
        
        # Send message
        return self.send_message(
            topic=KAFKA_ANOMALY_TOPIC,
            value=anomaly_data,
            key=device_id,
        )
    
    def send_telemetry(
        self,
        telemetry_data: Dict[str, Any],
        device_id: Optional[str] = None,
    ) -> bool:
        """
        Send telemetry data to the telemetry topic.
        
        Args:
            telemetry_data: Telemetry data
            device_id: Device ID (used as message key)
        
        Returns:
            True if message was sent, False otherwise
        """
        # Add timestamp if not present
        if "timestamp" not in telemetry_data:
            telemetry_data["timestamp"] = time.time()
        
        # Add metadata
        telemetry_data["metadata"] = {
            "source": "edgevision-guard",
            "version": "1.0.0",
            "device_id": device_id or os.getenv("DEVICE_ID", "unknown"),
        }
        
        # Send message
        return self.send_message(
            topic=KAFKA_TELEMETRY_TOPIC,
            value=telemetry_data,
            key=device_id,
        )
    
    def flush(self) -> None:
        """Flush the producer to ensure all messages are sent."""
        if self.enabled:
            self.producer.flush()
    
    def close(self) -> None:
        """Close the producer."""
        if self.enabled:
            self.producer.flush()
            logger.info("Kafka producer closed")


class KafkaConsumer:
    """Kafka consumer for receiving messages from Kafka topics."""
    
    def __init__(
        self,
        topics: List[str],
        group_id: str = KAFKA_GROUP_ID,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        security_protocol: str = KAFKA_SECURITY_PROTOCOL,
        sasl_mechanism: str = KAFKA_SASL_MECHANISM,
        username: str = KAFKA_USERNAME,
        password: str = KAFKA_PASSWORD,
        auto_offset_reset: str = "latest",
    ):
        """
        Initialize the Kafka consumer.
        
        Args:
            topics: List of Kafka topics to subscribe to
            group_id: Consumer group ID
            bootstrap_servers: Kafka bootstrap servers
            security_protocol: Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)
            sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512)
            username: SASL username
            password: SASL password
            auto_offset_reset: Auto offset reset strategy (earliest, latest)
        """
        # Check if Kafka is enabled
        if not KAFKA_ENABLED:
            logger.warning("Kafka integration is disabled. Set KAFKA_ENABLED=true to enable.")
            self.enabled = False
            return
        
        self.enabled = True
        self.topics = topics
        self.running = False
        self.consumer_thread = None
        
        # Configure Kafka consumer
        config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": True,
            "security.protocol": security_protocol,
        }
        
        # Add SASL configuration if using SASL
        if "SASL" in security_protocol:
            config.update({
                "sasl.mechanism": sasl_mechanism,
                "sasl.username": username,
                "sasl.password": password,
            })
        
        # Create Kafka consumer
        try:
            self.consumer = Consumer(config)
            logger.info(f"Kafka consumer initialized successfully for topics: {topics}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            self.enabled = False
            raise
    
    def start(self, message_handler: Callable[[str, Dict[str, Any], Optional[str]], None]) -> None:
        """
        Start consuming messages.
        
        Args:
            message_handler: Function to handle received messages (topic, value, key)
        """
        if not self.enabled:
            logger.warning("Kafka integration is disabled. Consumer not started.")
            return
        
        # Check if already running
        if self.running:
            logger.warning("Kafka consumer is already running")
            return
        
        # Subscribe to topics
        self.consumer.subscribe(self.topics)
        
        # Set running flag
        self.running = True
        
        # Define consumer loop
        def consumer_loop():
            while self.running:
                try:
                    # Poll for messages
                    msg = self.consumer.poll(timeout=1.0)
                    
                    # No message received
                    if msg is None:
                        continue
                    
                    # Error
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # End of partition event - not an error
                            logger.debug(f"Reached end of partition {msg.partition()}")
                        else:
                            # Error
                            logger.error(f"Kafka consumer error: {msg.error()}")
                    else:
                        # Message received
                        try:
                            # Get topic, key, and value
                            topic = msg.topic()
                            key = msg.key().decode("utf-8") if msg.key() else None
                            value = json.loads(msg.value().decode("utf-8"))
                            
                            # Call message handler
                            message_handler(topic, value, key)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                
                except KafkaException as e:
                    logger.error(f"Kafka exception: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    # Brief pause to avoid tight loop in case of persistent error
                    time.sleep(1)
        
        # Start consumer thread
        self.consumer_thread = threading.Thread(target=consumer_loop, daemon=True)
        self.consumer_thread.start()
        logger.info("Kafka consumer started")
    
    def stop(self) -> None:
        """Stop consuming messages."""
        if not self.enabled or not self.running:
            return
        
        # Set running flag to False
        self.running = False
        
        # Wait for consumer thread to finish
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5.0)
            self.consumer_thread = None
        
        # Close consumer
        self.consumer.close()
        logger.info("Kafka consumer stopped")


# Singleton instances
_producer: Optional[KafkaProducer] = None
_command_consumer: Optional[KafkaConsumer] = None


def get_kafka_producer() -> KafkaProducer:
    """
    Get a singleton Kafka producer instance.
    
    Returns:
        Kafka producer instance
    """
    global _producer
    if _producer is None:
        _producer = KafkaProducer()
    return _producer


def get_kafka_command_consumer(handler: Callable[[str, Dict[str, Any], Optional[str]], None]) -> KafkaConsumer:
    """
    Get a singleton Kafka command consumer instance.
    
    Args:
        handler: Function to handle received commands
    
    Returns:
        Kafka consumer instance
    """
    global _command_consumer
    if _command_consumer is None:
        _command_consumer = KafkaConsumer(topics=[KAFKA_COMMAND_TOPIC])
        _command_consumer.start(handler)
    return _command_consumer


def setup_kafka_integration(app: Optional[Any] = None) -> None:
    """
    Set up Kafka integration.
    
    Args:
        app: FastAPI application (optional)
    """
    if not KAFKA_ENABLED:
        logger.warning("Kafka integration is disabled. Set KAFKA_ENABLED=true to enable.")
        return
    
    # Initialize Kafka producer
    try:
        producer = get_kafka_producer()
        logger.info("Kafka producer initialized")
        
        # Send startup telemetry
        producer.send_telemetry({
            "event": "startup",
            "status": "online",
        })
    except Exception as e:
        logger.error(f"Failed to initialize Kafka producer: {e}")
    
    # Initialize Kafka command consumer if app is provided
    if app is not None:
        try:
            def command_handler(topic: str, value: Dict[str, Any], key: Optional[str]) -> None:
                """
                Handle received commands.
                
                Args:
                    topic: Kafka topic
                    value: Command payload
                    key: Message key
                """
                logger.info(f"Received command: {value}")
                
                # Handle different command types
                command_type = value.get("command")
                if command_type == "restart":
                    logger.info("Received restart command")
                    # Implement restart logic here
                elif command_type == "configure":
                    logger.info("Received configure command")
                    # Implement configuration update logic here
                elif command_type == "status":
                    logger.info("Received status request")
                    # Send status telemetry
                    producer = get_kafka_producer()
                    producer.send_telemetry({
                        "event": "status",
                        "status": "online",
                        "uptime": time.time() - app.state.startup_time,
                        "version": "1.0.0",
                    })
                else:
                    logger.warning(f"Unknown command type: {command_type}")
            
            # Start command consumer
            consumer = get_kafka_command_consumer(command_handler)
            logger.info("Kafka command consumer started")
            
            # Store consumer in app state for shutdown
            if app is not None:
                app.state.kafka_consumer = consumer
                
                # Add shutdown event handler
                @app.on_event("shutdown")
                def shutdown_kafka():
                    if hasattr(app.state, "kafka_consumer"):
                        app.state.kafka_consumer.stop()
                    
                    producer = get_kafka_producer()
                    producer.send_telemetry({
                        "event": "shutdown",
                        "status": "offline",
                    })
                    producer.close()
        except Exception as e:
            logger.error(f"Failed to initialize Kafka command consumer: {e}")
    
    logger.info("Kafka integration setup complete")


if __name__ == "__main__":
    # Test Kafka integration
    logging.basicConfig(level=logging.DEBUG)
    
    # Create producer
    producer = KafkaProducer()
    
    # Send test message
    producer.send_telemetry({
        "event": "test",
        "message": "Hello from EdgeVision-Guard!",
    })
    
    # Define command handler
    def handle_command(topic: str, value: Dict[str, Any], key: Optional[str]) -> None:
        print(f"Received command: {value}")
    
    # Create consumer
    consumer = KafkaConsumer(topics=[KAFKA_COMMAND_TOPIC])
    consumer.start(handle_command)
    
    # Wait for commands
    try:
        print("Waiting for commands (Ctrl+C to exit)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        consumer.stop()
        producer.close()