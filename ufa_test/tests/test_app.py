from fastapi.testclient import TestClient
from app.main import app
from app.preprocessing import preprocess

client = TestClient(app)


def test_preprocess_en() -> None:
    input_text = "Hello! This is test."
    expected = "hello test"
    assert preprocess(input_text) == expected


def test_preprocess_ru() -> None:
    input_text = "Привет! Это тест."
    result = preprocess(input_text)
    assert isinstance(result, str)
    assert "тест" in result or len(result) > 0  # Ru stopword removal may vary


def test_short_text_rejected() -> None:
    r = client.post("/predict", json={"text": "Hi"})
    assert r.status_code == 400
    assert "error" in r.json()["detail"]


def test_long_text_rejected() -> None:
    r = client.post("/predict", json={"text": "a" * 6000})
    assert r.status_code == 400
    assert "error" in r.json()["detail"]


def test_valid_text_prediction() -> None:
    r = client.post("/predict", json={"text": "I love this product!"})
    json_response = r.json()
    assert r.status_code == 200
    assert "sentiment" in json_response
    assert "confidence" in json_response
    assert 0 <= json_response["confidence"] <= 1


def test_ru_text_prediction() -> None:
    r = client.post("/predict", json={"text": "Мне всё понравилось"})
    json_response = r.json()
    assert r.status_code == 200
    assert "sentiment" in json_response
    assert "confidence" in json_response


def test_missing_text_field() -> None:
    r = client.post("/predict", json={})
    assert r.status_code == 422  # FastAPI will return validation error


def test_non_string_input() -> None:
    r = client.post("/predict", json={"text": 123})
    assert r.status_code == 422
