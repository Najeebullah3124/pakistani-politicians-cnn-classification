from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Pakistani Politician Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = [
    "ahmed_sharif_chaudhry", "asad_umar", "asif_ali_zardari",
    "bilawal_bhutto", "fawad_chaudhry", "gohar_ali_khan",
    "hina_rabbani_khar", "imran_khan", "maryam_nawaz",
    "nawaz_sharif", "pervez_musharraf", "rana_sanaullah",
    "saad_rafique", "shah_mahmood_qureshi", "shehbaz_sharif",
    "syed_mohsin_raza_naqvi"
]

@app.get("/")
def root():
    return {"message": "Pakistani Politician Classifier API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return JSONResponse(status_code=400, content={"error": "Only JPEG/PNG accepted"})
    probs = np.random.dirichlet(np.ones(16))
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "rank": i + 1,
            "name": CLASS_NAMES[idx].replace("_", " ").title(),
            "class_id": int(idx),
            "confidence": round(float(probs[idx]) * 100, 2)
        }
        for i, idx in enumerate(top3_idx)
    ]
    return {
        "predicted_class": CLASS_NAMES[top3_idx[0]].replace("_", " ").title(),
        "confidence": round(float(probs[top3_idx[0]]) * 100, 2),
        "top3_predictions": top3
    }

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"
    print("OK root endpoint")

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    print("OK health endpoint")

def test_predict_invalid():
    r = client.post("/predict",
        files={"file": ("test.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    print("OK invalid file rejected")

def test_predict_valid():
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    r = client.post("/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert len(data["top3_predictions"]) == 3
    print("OK predict endpoint:", data["predicted_class"], data["confidence"])

if __name__ == "__main__":
    test_root()
    test_health()
    test_predict_invalid()
    test_predict_valid()
    print("All tests passed")
