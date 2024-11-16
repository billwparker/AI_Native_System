import pytest
from fastapi.testclient import TestClient
from app import app, ResourceConstraints

@pytest.fixture
def client():
    return TestClient(app)

def test_create_task(client):
    response = client.post("/tasks", json={
        "intent": "example_task",
        "constraints": {
            "cpu_percent": 0.5,
            "memory_mb": 1024,
            "disk_mb": 512
        }
    })
    assert response.status_code == 200
    assert "task_id" in response.json()

def test_get_task(client):
    # First, create a task
    create_response = client.post("/tasks", json={
        "intent": "example_task",
        "constraints": {
            "cpu_percent": 0.5,
            "memory_mb": 1024,
            "disk_mb": 512
        }
    })
    assert create_response.status_code == 200
    task_id = create_response.json()["task_id"]

    # Then, retrieve the task
    get_response = client.get(f"/tasks/{task_id}")
    assert get_response.status_code == 200
    assert get_response.json()["intent"] == "example_task"

def test_get_nonexistent_task(client):
    response = client.get("/tasks/nonexistent_task_id")
    assert response.status_code == 422
    response_json = response.json()
    
    # Check structure and key elements without exact message matching
    assert "detail" in response_json
    assert isinstance(response_json["detail"], list)
    assert len(response_json["detail"]) == 1
    error = response_json["detail"][0]
    
    # Verify key error properties
    assert error["type"] == "uuid_parsing"  # Updated error type
    assert error["loc"] == ["path", "task_id"]
    assert "invalid" in error["msg"].lower()
    assert "uuid" in error["msg"].lower()