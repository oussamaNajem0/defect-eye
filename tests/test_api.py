import pytest
from fastapi.testclient import TestClient
from defect_eye.api import app

client = TestClient(app)

def test_health_check_no_model(mocker):
    """Test health check when model fails to load."""
    # Mock the global MODEL to be None
    mocker.patch("defect_eye.api.MODEL", None)
    response = client.get("/health")
    assert response.status_code == 503

def test_health_check_with_model(mocker):
    """Test health check when model is loaded."""
    mocker.patch("defect_eye.api.MODEL", "mock_model")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_status": "loaded"}

def test_predict_endpoint_success(mocker):
    """Test successful prediction response."""
    # Create a mock model class
    class MockModel:
        def predict_proba(self, X):
            return [[0.1, 0.85]]
        def predict(self, X):
            return [1]
            
    mocker.patch("defect_eye.api.MODEL", MockModel())
    
    payload = {
        "loc": 250.0,
        "cyclomatic_complexity": 22.0,
        "halstead_volume": 1050.0
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["defect_probability"] == 0.85

def test_predict_endpoint_validation_error():
    """Test API rejection of incomplete payloads."""
    # Missing required 'loc' field
    payload = {
        "cyclomatic_complexity": 22.0,
        "halstead_volume": 1050.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # FastAPI validation error