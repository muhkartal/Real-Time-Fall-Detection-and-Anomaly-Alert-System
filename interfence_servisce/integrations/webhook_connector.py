#!/usr/bin/env python3
"""
Webhook integration for EdgeVision-Guard.

This module provides webhook notification capabilities for integration with external systems.
"""

import asyncio
import hmac
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urljoin

import aiohttp
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
WEBHOOK_ENABLED = os.getenv("WEBHOOK_ENABLED", "false").lower() in ("true", "1", "yes")
WEBHOOK_ENDPOINTS = os.getenv("WEBHOOK_ENDPOINTS", "").split(",")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", "5"))
WEBHOOK_RETRY_COUNT = int(os.getenv("WEBHOOK_RETRY_COUNT", "3"))
WEBHOOK_RETRY_DELAY = int(os.getenv("WEBHOOK_RETRY_DELAY", "2"))
WEBHOOK_ALERT_THRESHOLD = float(os.getenv("WEBHOOK_ALERT_THRESHOLD", "0.7"))


class WebhookPayload(BaseModel):
    """Base model for webhook payloads."""
    
    event_type: str = Field(..., description="Type of event")
    event_id: str = Field(..., description="Unique event ID")
    timestamp: float = Field(..., description="Event timestamp")
    source: str = Field(..., description="Event source")
    device_id: str = Field(..., description="Device ID")
    data: Dict[str, Any] = Field(..., description="Event data")
    version: str = Field(..., description="API version")


class WebhookConfig(BaseModel):
    """Configuration for a webhook endpoint."""
    
    url: str = Field(..., description="Webhook URL")
    secret: Optional[str] = Field(None, description="Webhook secret for HMAC signature")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    event_types: Set[str] = Field(default_factory=set, description="Event types to send")
    enabled: bool = Field(True, description="Whether this webhook is enabled")
    version: str = Field("1.0", description="Webhook API version")
    retry_count: int = Field(WEBHOOK_RETRY_COUNT, description="Number of retries")
    retry_delay: int = Field(WEBHOOK_RETRY_DELAY, description="Delay between retries in seconds")
    timeout: int = Field(WEBHOOK_TIMEOUT, description="Request timeout in seconds")


class WebhookManager:
    """Manager for webhook notifications."""
    
    def __init__(self):
        """Initialize the webhook manager."""
        self.configs: List[WebhookConfig] = []
        self.client = httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT)
        self.enabled = WEBHOOK_ENABLED
        
        # Load configurations
        self._load_configs()
        
        logger.info(f"Webhook manager initialized with {len(self.configs)} configurations")
    
    def _load_configs(self) -> None:
        """Load webhook configurations from environment variables."""
        if not self.enabled:
            logger.warning("Webhook integration is disabled. Set WEBHOOK_ENABLED=true to enable.")
            return
        
        try:
            # Load from WEBHOOK_ENDPOINTS environment variable
            for endpoint in WEBHOOK_ENDPOINTS:
                if not endpoint.strip():
                    continue
                
                # Default configuration
                config = WebhookConfig(
                    url=endpoint.strip(),
                    secret=WEBHOOK_SECRET,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "EdgeVision-Guard/1.0",
                    },
                    event_types={"anomaly", "alert", "status"},
                )
                
                self.configs.append(config)
            
            # Load from config file if exists
            config_path = os.getenv("WEBHOOK_CONFIG_FILE", "config/webhooks.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                
                for webhook_config in config_data.get("webhooks", []):
                    config = WebhookConfig(**webhook_config)
                    self.configs.append(config)
            
            logger.info(f"Loaded {len(self.configs)} webhook configurations")
        except Exception as e:
            logger.error(f"Error loading webhook configurations: {e}")
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """
        Generate HMAC signature for payload.
        
        Args:
            payload: JSON payload
            secret: Secret key
        
        Returns:
            HMAC signature
        """
        return hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
    
    async def send_webhook(self, config: WebhookConfig, payload: Dict[str, Any]) -> bool:
        """
        Send webhook notification.
        
        Args:
            config: Webhook configuration
            payload: Event payload
        
        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not config.enabled:
            return False
        
        # Check if event type is in allowed types
        if config.event_types and payload.get("event_type") not in config.event_types:
            return False
        
        # Serialize payload to JSON
        payload_json = json.dumps(payload)
        
        # Set up headers
        headers = config.headers.copy()
        
        # Add signature if secret is provided
        if config.secret:
            signature = self._generate_signature(payload_json, config.secret)
            headers["X-EdgeVision-Signature"] = f"sha256={signature}"
        
        # Send webhook notification with retries
        for attempt in range(config.retry_count + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        config.url,
                        data=payload_json,
                        headers=headers,
                        timeout=config.timeout,
                    ) as response:
                        if response.status < 400:
                            logger.info(f"Webhook notification sent successfully to {config.url}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"Webhook notification failed (attempt {attempt + 1}/{config.retry_count + 1}): "
                                f"Status {response.status}, Response: {error_text}"
                            )
            except Exception as e:
                logger.warning(
                    f"Webhook notification failed (attempt {attempt + 1}/{config.retry_count + 1}): {e}"
                )
            
            # Retry with delay if not the last attempt
            if attempt < config.retry_count:
                await asyncio.sleep(config.retry_delay)
        
        logger.error(f"Webhook notification failed after {config.retry_count + 1} attempts")
        return False
    
    async def send_event(self, event_type: str, data: Dict[str, Any], device_id: Optional[str] = None) -> bool:
        """
        Send an event notification to all configured webhooks.
        
        Args:
            event_type: Type of event
            data: Event data
            device_id: Device ID
        
        Returns:
            True if at least one notification was sent successfully, False otherwise
        """
        if not self.enabled or not self.configs:
            logger.warning("Webhook integration is disabled or no configurations found")
            return False
        
        # Create payload
        payload = WebhookPayload(
            event_type=event_type,
            event_id=f"{int(time.time() * 1000)}-{os.getpid()}",
            timestamp=time.time(),
            source="edgevision-guard",
            device_id=device_id or os.getenv("DEVICE_ID", "unknown"),
            data=data,
            version="1.0",
        ).dict()
        
        # Send to all configured webhooks
        results = await asyncio.gather(*[
            self.send_webhook(config, payload)
            for config in self.configs
        ])
        
        return any(results)
    
    async def send_anomaly(self, anomaly_data: Dict[str, Any], device_id: Optional[str] = None) -> bool:
        """
        Send an anomaly detection notification.
        
        Args:
            anomaly_data: Anomaly data
            device_id: Device ID
        
        Returns:
            True if notification was sent successfully, False otherwise
        """
        return await self.send_event("anomaly", anomaly_data, device_id)
    
    async def send_alert(self, alert_data: Dict[str, Any], device_id: Optional[str] = None) -> bool:
        """
        Send an alert notification.
        
        Args:
            alert_data: Alert data
            device_id: Device ID
        
        Returns:
            True if notification was sent successfully, False otherwise
        """
        return await self.send_event("alert", alert_data, device_id)
    
    async def send_status(self, status_data: Dict[str, Any], device_id: Optional[str] = None) -> bool:
        """
        Send a status notification.
        
        Args:
            status_data: Status data
            device_id: Device ID
        
        Returns:
            True if notification was sent successfully, False otherwise
        """
        return await self.send_event("status", status_data, device_id)
    
    async def close(self) -> None:
        """Close the webhook manager."""
        await self.client.aclose()
        logger.info("Webhook manager closed")


# Singleton instance
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """
    Get a singleton webhook manager instance.
    
    Returns:
        Webhook manager instance
    """
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    
    return _webhook_manager


async def send_prediction_webhook(prediction_result: Dict[str, Any]) -> bool:
    """
    Send a prediction result as a webhook notification.
    
    Args:
        prediction_result: Prediction result
    
    Returns:
        True if notification was sent successfully, False otherwise
    """
    # Check if this is an anomaly
    anomaly_score = prediction_result.get("anomaly_score", 0.0)
    
    if anomaly_score >= WEBHOOK_ALERT_THRESHOLD:
        # Send as alert
        manager = get_webhook_manager()
        
        # Create alert data
        alert_data = {
            "prediction": prediction_result.get("prediction", 0),
            "class_name": prediction_result.get("class_name", "Unknown"),
            "confidence": prediction_result.get("confidence", 0.0),
            "anomaly_score": anomaly_score,
            "timestamp": prediction_result.get("timestamp", time.time()),
            "severity": "high" if anomaly_score > 0.9 else "medium",
            "location": os.getenv("DEVICE_LOCATION", "unknown"),
        }
        
        return await manager.send_alert(alert_data)
    elif anomaly_score >= 0.5:
        # Send as anomaly
        manager = get_webhook_manager()
        return await manager.send_anomaly(prediction_result)
    
    return False


def setup_webhook_integration(app=None) -> None:
    """
    Set up webhook integration.
    
    Args:
        app: FastAPI application (optional)
    """
    if not WEBHOOK_ENABLED:
        logger.warning("Webhook integration is disabled. Set WEBHOOK_ENABLED=true to enable.")
        return
    
    # Initialize webhook manager
    manager = get_webhook_manager()
    
    if app:
        # Store manager in app state
        app.state.webhook_manager = manager
        
        # Send startup status
        @app.on_event("startup")
        async def startup_webhook():
            await manager.send_status({
                "status": "online",
                "version": "1.0.0",
                "startup_time": time.time(),
            })
        
        # Close manager on shutdown
        @app.on_event("shutdown")
        async def shutdown_webhook():
            await manager.send_status({
                "status": "offline",
                "shutdown_time": time.time(),
            })
            await manager.close()
    
    logger.info("Webhook integration setup complete")


if __name__ == "__main__":
    # Test webhook integration
    async def test_webhooks():
        # Initialize webhook manager
        manager = get_webhook_manager()
        
        # Send test alert
        result = await manager.send_alert({
            "message": "Test alert",
            "severity": "high",
            "anomaly_score": 0.95,
        })
        
        print(f"Alert sent: {result}")
        
        # Send test status
        result = await manager.send_status({
            "status": "online",
            "uptime": 3600,
        })
        
        print(f"Status sent: {result}")
        
        # Clean up
        await manager.close()
    
    asyncio.run(test_webhooks())