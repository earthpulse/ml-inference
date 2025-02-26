import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys

from api.main import app

client = TestClient(app)

@pytest.fixture
def client_with_api_key():
    """Create a test client with API key set"""
    # Save original modules
    original_modules = sys.modules.copy()
    
    # Set environment variable
    os.environ["API_KEY"] = "test_key"
    
    # Force reload of the main module
    if "api.main" in sys.modules:
        del sys.modules["api.main"]
    
    # Import the module again
    from api.main import app
    client = TestClient(app)
    
    yield client
    
    # Restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)
    os.environ.pop("API_KEY", None)

@pytest.fixture
def client_without_api_key():
    """Create a test client with no API key set"""
    # Save original modules
    original_modules = sys.modules.copy()
    
    # Set environment variable
    os.environ.pop("API_KEY", None)
    
    # Import the module again
    from api.main import app
    client = TestClient(app)    
    
    yield client
    
    # Restore original modulesc
    sys.modules.clear()
    sys.modules.update(original_modules)
    os.environ.pop("API_KEY", None)

def test_hello_endpoint_no_auth(client_without_api_key):
    """Test the hello endpoint when no API key is required"""
    response = client_without_api_key.get("/")
    assert response.status_code == 200
    assert response.json()["auth_required"] is False

def test_hello_endpoint_with_auth(client_with_api_key):
    """Test the hello endpoint when API key is required"""
    response = client_with_api_key.get("/")
    assert response.status_code == 200
    assert response.json()["auth_required"] is True

# def test_model_endpoint_no_auth(client_without_api_key):
#     """Test model endpoint when no API key is required"""
#     with patch("api.main.ModelWrapper") as mock_model:
#         mock_instance = mock_model.return_value
#         mock_instance.gdf.to_json.return_value = '{"test": "data"}'
#         response = client_without_api_key.get("/test_model")
#         assert response.status_code == 200

def test_model_endpoint_with_auth_missing_key(client_with_api_key):
    """Test model endpoint when API key is required but not provided"""
    response = client_with_api_key.get("/test_model")
    assert response.status_code == 401
    assert "API Key is required" in response.json()["detail"]

def test_model_endpoint_with_auth_invalid_key(client_with_api_key):
    """Test model endpoint when API key is required but invalid key is provided"""
    response = client_with_api_key.get("/test_model", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 403
    assert "Invalid API Key" in response.json()["detail"]

def test_model_endpoint_with_auth_valid_key(client_with_api_key):
    """Test model endpoint when API key is required and valid key is provided"""
    # Mock the ModelWrapper to avoid actual model loading
    with patch("api.main.ModelWrapper") as mock_model:
        mock_instance = mock_model.return_value
        mock_instance.gdf.to_json.return_value = '{"test": "data"}'
        response = client_with_api_key.get("/test_model", headers={"X-API-Key": "test_key"})
        assert response.status_code == 200

# def test_inference_endpoint_auth(client_with_api_key):
#     """Basic test for inference endpoint authentication"""
#     # This is a simplified test that just checks auth, not the full functionality
#     response = client_with_api_key.post("/test_model", files={"image": ("test.tif", b"dummy data")})
#     assert response.status_code == 401
    
#     # With valid key
#     with patch("api.main.ModelWrapper"), \
#          patch("api.main.process_in_batch"), \
#          patch("io.BytesIO"), \
#          patch("rasterio.open"):
#         # This would need more mocking for a complete test
#         # Just testing that auth mechanism works
#         response = client_with_api_key.post(
#             "/test_model", 
#             files={"image": ("test.tif", b"dummy data")},
#             headers={"X-API-Key": "test_key"}
#         )
#         # We'll get a different error than 401 if auth passes
#         assert response.status_code != 401