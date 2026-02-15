from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post(
        "/predict",
        json={"features": [10.0, 1.0]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["status"] == "success"

def test_upload_csv():
    # Create a temporary CSV
    csv_content = "feature1,feature2\n10,1\n2,5"
    files = {'file': ('test.csv', csv_content, 'text/csv')}
    response = client.post("/upload/csv", files=files)
    assert response.status_code == 200
    assert response.json()["filename"] == "test.csv"
    assert response.json()["rows"] == 2

def test_upload_txt():
    txt_content = "Hello World"
    files = {'file': ('test.txt', txt_content, 'text/plain')}
    response = client.post("/upload/txt", files=files)
    assert response.status_code == 200
    assert "char_count" in response.json()

def test_upload_invalid_file():
    files = {'file': ('test.exe', b'something', 'application/octet-stream')}
    response = client.post("/upload/csv", files=files)
    assert response.status_code == 400
