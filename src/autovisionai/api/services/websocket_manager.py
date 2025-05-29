import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manager for WebSocket connections and broadcasting."""

    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: List[WebSocket] = []
        logger.info("WebSocket manager initialized")

    async def connect(self, websocket: WebSocket) -> None:
        """Connect a new WebSocket client.

        Args:
            websocket (WebSocket): The WebSocket connection to add
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection added. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket client.

        Args:
            websocket (WebSocket): The WebSocket connection to remove
        """
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection removed. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message (Dict[str, Any]): Message to broadcast
        """
        if not self.active_connections:
            logger.debug("No active connections to broadcast to")
            return

        logger.info(f"Broadcasting message to {len(self.active_connections)} connections: {message}")
        disconnect = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                logger.info("Successfully sent message to connection")
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {str(e)}", exc_info=True)
                # Remove failed connection
                disconnect.append(connection)
                logger.info(f"Removed failed connection. Total connections: {len(self.active_connections)}")

        for connection in disconnect:
            await self.disconnect(connection)
