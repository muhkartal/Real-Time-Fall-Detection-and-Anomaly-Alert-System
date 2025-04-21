#!/usr/bin/env python3
"""
MQTT integration for EdgeVision-Guard.

This module provides an MQTT client for IoT and edge device communication.
"""

import json
import logging
import os
import ssl
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

import paho.mqtt.client as mqtt
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
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", f"edgevision-guard-{os.getpid()}")
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_USE_TLS = os.getenv("MQTT_USE_TLS", "false").lower() in ("true", "1", "yes")
MQTT_TLS_CA_CERTS = os.getenv("MQTT_TLS_CA_CERTS", "")
MQTT_QOS = int(os.getenv("MQTT_QOS", "1"))
MQTT_RETAIN = os.getenv("MQTT_RETAIN", "false").lower() in ("true", "1", "yes")
MQTT_ENABLED = os.getenv("MQTT_ENABLED", "false").lower() in ("true", "1", "yes")

# Topics
MQTT_BASE_TOPIC = os.getenv("MQTT_BASE_TOPIC", "edgevision/")
MQTT_ANOMALY_TOPIC = f"{MQTT_BASE_TOPIC}anomalies"
MQTT_TELEMETRY_TOPIC = f"{MQTT_BASE_TOPIC}telemetry"
MQTT_COMMAND_TOPIC = f"{MQTT_BASE_TOPIC}commands"
MQTT_STATUS_TOPIC = f"{MQTT_BASE_TOPIC}status"


class MQTTClient:
    """MQTT client for IoT and edge device communication."""
    
    def __init__(
        self,
        client_id: str = MQTT_CLIENT_ID,
        broker_host: str = MQTT_BROKER_HOST,
        broker_port: int = MQTT_BROKER_PORT,
        username: str = MQTT_USERNAME,
        password: str = MQTT_PASSWORD,
        use_tls: bool = MQTT_USE_TLS,
        ca_certs: str = MQTT_TLS_CA_CERTS,
        qos: int = MQTT_QOS,
        retain: bool = MQTT_RETAIN,
    ):
        """
        Initialize the MQTT client.
        
        Args:
            client_id: MQTT client ID
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            username: MQTT username
            password: MQTT password
            use_tls: Whether to use TLS
            ca_certs: Path to CA certificate file
            qos: MQTT QoS level (0, 1, or 2)
            retain: Whether to retain messages
        """
        # Check if MQTT is enabled
        if not MQTT_ENABLED:
            logger.warning("MQTT integration is disabled. Set MQTT_ENABLED=true to enable.")
            self.enabled = False
            return
        
        self.enabled = True
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.ca_certs = ca_certs
        self.qos = qos
        self.retain = retain
        
        # Keep track of subscriptions and handlers
        self.subscriptions = {}
        self.connected = False
        self.connection_error = None
        
        # Create MQTT client
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)
        
        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Set up authentication if provided
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Set up TLS if enabled
        if use_tls:
            if ca_certs:
                self.client.tls_set(
                    ca_certs=ca_certs,
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLS,
                )
            else:
                self.client.tls_set(
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLS,
                )
            self.client.tls_insecure_set(False)
        
        # Initialize status reporting
        self.status_thread = None
        self.status_interval = 60  # seconds
        self.running = False
        
        logger.info(f"MQTT client initialized for broker {broker_host}:{broker_port}")
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """
        Callback for when the client connects to the broker.
        
        Args:
            client: MQTT client
            userdata: User data
            flags: Connection flags
            rc: Result code
            properties: Protocol properties (MQTTv5)
        """
        if rc == 0:
            logger.info(f"Connected to MQTT broker {self.broker_host}:{self.broker_port}")
            self.connected = True
            self.connection_error = None
            
            # Resubscribe to topics
            for topic, callback in self.subscriptions.items():
                client.subscribe(topic, qos=self.qos)
                logger.info(f"Resubscribed to topic: {topic}")
            
            # Publish online status
            self.publish_status("online")
        else:
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorized",
            }
            error_message = error_messages.get(rc, f"Unknown error code: {rc}")
            logger.error(f"Failed to connect to MQTT broker: {error_message}")
            self.connected = False
            self.connection_error = error_message
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """
        Callback for when the client disconnects from the broker.
        
        Args:
            client: MQTT client
            userdata: User data
            rc: Result code
            properties: Protocol properties (MQTTv5)
        """
        if rc == 0:
            logger.info("Disconnected from MQTT broker")
        else:
            logger.warning(f"Unexpected disconnection from MQTT broker (code: {rc})")
        
        self.connected = False
    
    def _on_message(self, client, userdata, msg):
        """
        Callback for when a message is received from the broker.
        
        Args:
            client: MQTT client
            userdata: User data
            msg: Received message
        """
        try:
            # Find a matching subscription
            for topic_pattern, callback in self.subscriptions.items():
                if mqtt.topic_matches_sub(topic_pattern, msg.topic):
                    # Parse message payload as JSON
                    try:
                        payload = json.loads(msg.payload.decode("utf-8"))
                    except json.JSONDecodeError:
                        payload = msg.payload.decode("utf-8")
                    
                    # Call callback
                    callback(msg.topic, payload)
                    break
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def connect(self) -> bool:
        """
        Connect to the MQTT broker.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.enabled:
            logger.warning("MQTT integration is disabled. Cannot connect.")
            return False
        
        try:
            # Set last will (status message)
            self.client.will_set(
                topic=MQTT_STATUS_TOPIC,
                payload=json.dumps({
                    "status": "offline",
                    "device_id": os.getenv("DEVICE_ID", "unknown"),
                    "timestamp": time.time(),
                }),
                qos=self.qos,
                retain=self.retain,
            )
            
            # Connect to broker
            self.client.connect(
                host=self.broker_host,
                port=self.broker_port,
                keepalive=60,
            )
            
            # Start the network loop in a separate thread
            self.client.loop_start()
            
            # Wait for connection or error
            for _ in range(10):
                if self.connected:
                    return True
                if self.connection_error:
                    logger.error(f"Connection failed: {self.connection_error}")
                    return False
                time.sleep(0.5)
            
            logger.error("Connection timed out")
            return False
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if not self.enabled or not hasattr(self, "client"):
            return
        
        # Stop status thread
        self.running = False
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=2.0)
        
        # Publish offline status
        if self.connected:
            self.publish_status("offline")
        
        # Disconnect from broker
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")
    
    def subscribe(self, topic: str, callback: Callable[[str, Any], None]) -> bool:
        """
        Subscribe to a topic.
        
        Args:
            topic: MQTT topic
            callback: Callback function (topic, payload)
        
        Returns:
            True if subscribed, False otherwise
        """
        if not self.enabled:
            logger.warning("MQTT integration is disabled. Cannot subscribe.")
            return False
        
        try:
            # Add subscription to dictionary
            self.subscriptions[topic] = callback
            
            # Subscribe to topic if connected
            if self.connected:
                result, _ = self.client.subscribe(topic, qos=self.qos)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Subscribed to topic: {topic}")
                    return True
                else:
                    logger.error(f"Failed to subscribe to topic: {topic}")
                    return False
            
            return False
        except Exception as e:
            logger.error(f"Error subscribing to topic: {e}")
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: MQTT topic
        
        Returns:
            True if unsubscribed, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Remove subscription from dictionary
            if topic in self.subscriptions:
                del self.subscriptions[topic]
            
            # Unsubscribe from topic if connected
            if self.connected:
                result, _ = self.client.unsubscribe(topic)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Unsubscribed from topic: {topic}")
                    return True
                else:
                    logger.error(f"Failed to unsubscribe from topic: {topic}")
                    return False
            
            return False
        except Exception as e:
            logger.error(f"Error unsubscribing from topic: {e}")
            return False
    
    def publish(self, topic: str, payload: Any, qos: Optional[int] = None, retain: Optional[bool] = None) -> bool:
        """
        Publish a message to a topic.
        
        Args:
            topic: MQTT topic
            payload: Message payload (will be JSON serialized if a dict)
            qos: QoS level (0, 1, or 2)
            retain: Whether to retain the message
        
        Returns:
            True if published, False otherwise
        """
        if not self.enabled:
            logger.warning("MQTT integration is disabled. Message not published.")
            return False
        
        if not self.connected:
            logger.warning("Not connected to MQTT broker. Message not published.")
            return False
        
        try:
            # Convert payload to JSON if it's a dict
            if isinstance(payload, dict):
                payload = json.dumps(payload)
            
            # Set QoS and retain flags
            qos = qos if qos is not None else self.qos
            retain = retain if retain is not None else self.retain
            
            # Publish message
            result = self.client.publish(
                topic=topic,
                payload=payload,
                qos=qos,
                retain=retain,
            )
            
            # Check result
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published message to topic: {topic}")
                return True
            else:
                logger.error(f"Failed to publish message to topic: {topic}")
                return False
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    def publish_anomaly(self, anomaly_data: Dict[str, Any]) -> bool:
        """
        Publish an anomaly detection.
        
        Args:
            anomaly_data: Anomaly data
        
        Returns:
            True if published, False otherwise
        """
        # Add timestamp if not present
        if "timestamp" not in anomaly_data:
            anomaly_data["timestamp"] = time.time()
        
        # Add device ID if not present
        if "device_id" not in anomaly_data:
            anomaly_data["device_id"] = os.getenv("DEVICE_ID", "unknown")
        
        # Publish message
        return self.publish(MQTT_ANOMALY_TOPIC, anomaly_data)
    
    def publish_telemetry(self, telemetry_data: Dict[str, Any]) -> bool:
        """
        Publish telemetry data.
        
        Args:
            telemetry_data: Telemetry data
        
        Returns:
            True if published, False otherwise
        """
        # Add timestamp if not present
        if "timestamp" not in telemetry_data:
            telemetry_data["timestamp"] = time.time()
        
        # Add device ID if not present
        if "device_id" not in telemetry_data:
            telemetry_data["device_id"] = os.getenv("DEVICE_ID", "unknown")
        
        # Publish message
        return self.publish(MQTT_TELEMETRY_TOPIC, telemetry_data)
    
    def publish_status(self, status: str) -> bool:
        """
        Publish device status.
        
        Args:
            status: Status string ("online" or "offline")
        
        Returns:
            True if published, False otherwise
        """
        # Create status message
        status_data = {
            "status": status,
            "device_id": os.getenv("DEVICE_ID", "unknown"),
            "timestamp": time.time(),
            "version": "1.0.0",
        }
        
        # Publish message with retain flag
        return self.publish(MQTT_STATUS_TOPIC, status_data, retain=True)
    
    def start_status_reporting(self, interval: int = 60) -> None:
        """
        Start periodic status reporting.
        
        Args:
            interval: Reporting interval in seconds
        """
        if not self.enabled:
            return
        
        self.status_interval = interval
        self.running = True
        
        # Define status reporting thread
        def status_loop():
            while self.running:
                if self.connected:
                    # Publish online status
                    self.publish_status("online")
                    
                    # Publish system stats
                    import psutil
                    stats = {
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage("/").percent,
                    }
                    self.publish_telemetry({
                        "type": "system_stats",
                        "stats": stats,
                    })
                
                # Sleep for the specified interval
                for _ in range(self.status_interval):
                    if not self.running:
                        break
                    time.sleep(1)
        
        # Start status reporting thread
        self.status_thread = threading.Thread(target=status_loop, daemon=True)
        self.status_thread.start()
        logger.info(f"Status reporting started with interval {interval} seconds")
    
    def setup_command_handler(self, command_handler: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Set up a handler for commands.
        
        Args:
            command_handler: Function to handle received commands
        
        Returns:
            True if subscription was successful, False otherwise
        """
        def on_command(topic, payload):
            try:
                if isinstance(payload, str):
                    payload = json.loads(payload)
                
                # Call command handler
                command_handler(payload)
            except Exception as e:
                logger.error(f"Error handling command: {e}")
        
        # Subscribe to command topic
        return self.subscribe(f"{MQTT_COMMAND_TOPIC}/#", on_command)


# Singleton instance
_mqtt_client: Optional[MQTTClient] = None


def get_mqtt_client() -> MQTTClient:
    """
    Get a singleton MQTT client instance.
    
    Returns:
        MQTT client instance
    """
    global _mqtt_client
    if _mqtt_client is None:
        _mqtt_client = MQTTClient()
    
    return _mqtt_client


def setup_mqtt_integration(app=None) -> bool:
    """
    Set up MQTT integration.
    
    Args:
        app: FastAPI application (optional)
    
    Returns:
        True if setup was successful, False otherwise
    """
    if not MQTT_ENABLED:
        logger.warning("MQTT integration is disabled. Set MQTT_ENABLED=true to enable.")
        return False
    
    try:
        # Get MQTT client
        client = get_mqtt_client()
        
        # Connect to broker
        if not client.connect():
            logger.error("Failed to connect to MQTT broker")
            return False
        
        # Set up command handler if app is provided
        if app is not None:
            def command_handler(command):
                logger.info(f"Received command: {command}")
                
                # Handle different command types
                command_type = command.get("command")
                if command_type == "restart":
                    logger.info("Received restart command")
                    # Implement restart logic here
                elif command_type == "configure":
                    logger.info("Received configure command")
                    # Implement configuration update logic here
                elif command_type == "status":
                    logger.info("Received status request")
                    # Publish status
                    client.publish_status("online")
                else:
                    logger.warning(f"Unknown command type: {command_type}")
            
            # Set up command handler
            client.setup_command_handler(command_handler)
            
            # Store client in app state
            app.state.mqtt_client = client
            
            # Add shutdown event handler
            @app.on_event("shutdown")
            def shutdown_mqtt():
                logger.info("Shutting down MQTT client")
                if hasattr(app.state, "mqtt_client"):
                    app.state.mqtt_client.disconnect()
        
        # Start status reporting
        client.start_status_reporting()
        
        logger.info("MQTT integration setup complete")
        return True
    except Exception as e:
        logger.error(f"Error setting up MQTT integration: {e}")
        return False


if __name__ == "__main__":
    # Test MQTT integration
    logging.basicConfig(level=logging.DEBUG)
    
    # Create client
    client = MQTTClient()
    
    # Connect to broker
    if client.connect():
        # Set up command handler
        def handle_command(command):
            print(f"Received command: {command}")
        
        client.setup_command_handler(handle_command)
        
        # Start status reporting
        client.start_status_reporting(interval=10)
        
        # Publish test telemetry
        client.publish_telemetry({
            "type": "test",
            "message": "Hello from EdgeVision-Guard!",
        })
        
        # Wait for commands
        try:
            print("Waiting for commands (Ctrl+C to exit)...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            client.disconnect()
    else:
        print("Failed to connect to MQTT broker")