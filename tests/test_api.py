"""Tests against the real FastAPI app in app/main.py."""

import io
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "app"))

from main import app  # noqa: E402

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_api_root():
    r = client.get("/api")
    assert r.status_code == 200
    assert r.json().get("status") == "running"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_invalid():
    r = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert r.status_code in (400, 503)


def test_predict_valid():
    h = client.get("/health").json()
    if not h.get("model_loaded"):
        pytest.skip("Model not loaded (expected in CI without weights)")

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert len(data["top3_predictions"]) == 3
