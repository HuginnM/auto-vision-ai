import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from autovisionai.api.main import app


@pytest.fixture
def client():
    """Create a test client for FastAPI application."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client for FastAPI application."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
