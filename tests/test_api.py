import io
import os
import pytest
import torch
import torch.nn as nn

# Set environment variables to control startup behaviors during tests
os.environ["MODEL_NAME"] = "cnn"
os.environ["MODEL_WEIGHTS_PATH"] = "dummy.pt"

from src.app.main import create_app

class MockClassificationModel(nn.Module):
    """
    Mock classification model to bypass heavy pytorch weight loading
    and network initializations during test runs.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 39)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        # Force class 0 (Apple Scab) to have high score for predictive consistency
        logits[:, 0] = 50.0
        return logits


@pytest.fixture
def app():
    # Setup test Flask application context
    app = create_app()
    app.config.update({
        "TESTING": True,
        "MODEL": MockClassificationModel()
    })
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_health_check_endpoint(client):
    """Verifies standard server status and metadata report."""
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data['status'] == 'healthy'
    assert 'model' in res_data
    assert res_data['model']['name'] == 'cnn'


def test_get_diseases_metadata(client):
    """Verifies lists containing all 39 standard categories."""
    response = client.get('/api/v1/diseases')
    assert response.status_code == 200
    res_data = response.get_json()
    assert 'diseases' in res_data
    assert len(res_data['diseases']) == 39
    assert res_data['diseases'][0]['class_id'] == 0


def test_get_disease_by_id(client):
    """Verifies query returns matching index descriptions."""
    response = client.get('/api/v1/diseases/0')
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data['class_id'] == 0
    assert 'disease_name' in res_data
    assert 'Apple' in res_data['disease_name']


def test_get_disease_by_id_out_of_bounds(client):
    """Verifies that queries exceeding valid bounds trigger 404."""
    response = client.get('/api/v1/diseases/999')
    assert response.status_code == 404
    res_data = response.get_json()
    assert 'error' in res_data


def test_predict_missing_image_payload(client):
    """Verifies requests without file payloads trigger 400 validation."""
    response = client.post('/api/v1/predict')
    assert response.status_code == 400
    res_data = response.get_json()
    assert 'error' in res_data
    assert 'No image file provided' in res_data['error']


def test_predict_invalid_file_extension(client):
    """Verifies that text/doc inputs are rejected by file parsing rules."""
    payload = {
        'image': (io.BytesIO(b"mock byte data content"), 'invalid_doc.txt')
    }
    response = client.post('/api/v1/predict', data=payload, content_type='multipart/form-data')
    assert response.status_code == 400
    res_data = response.get_json()
    assert 'error' in res_data
    assert 'Invalid file extension' in res_data['error']


def test_metrics_endpoint(client):
    """Verifies Prometheus exposition format contains necessary telemetry strings."""
    response = client.get('/metrics')
    assert response.status_code == 200
    assert b"plant_disease_predictions_total" in response.data
    assert b"plant_disease_uptime_seconds" in response.data
