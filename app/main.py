
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Pakistani Politician Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Class names (must match training order) 
CLASS_NAMES = [
    "ahmed_sharif_chaudhry", "asad_umar", "asif_ali_zardari",
    "bilawal_bhutto", "fawad_chaudhry", "gohar_ali_khan",
    "hina_rabbani_khar", "imran_khan", "maryam_nawaz",
    "nawaz_sharif", "pervez_musharraf", "rana_sanaullah",
    "saad_rafique", "shah_mahmood_qureshi", "shehbaz_sharif",
    "syed_mohsin_raza_naqvi"
]

# ── Load model once at startup
MODEL_PATH = "resnet_final.keras"   # ResNet50 — best test accuracy
model      = None

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    # ResNet50 preprocessing (mean subtraction)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

@app.get("/")
def root():
    return {"message": "Pakistani Politician Classifier API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Only JPEG/PNG images accepted"}
        )

    image_bytes = await file.read()
    img_array   = preprocess_image(image_bytes)
    predictions = model.predict(img_array, verbose=0)[0]

    # Top 3 predictions
    top3_idx  = np.argsort(predictions)[::-1][:3]
    top3      = [
        {
            "rank":       int(i + 1),
            "name":       CLASS_NAMES[idx].replace("_", " ").title(),
            "class_id":   int(idx),
            "confidence": round(float(predictions[idx]) * 100, 2)
        }
        for i, idx in enumerate(top3_idx)
    ]

    return {
        "predicted_class":   CLASS_NAMES[top3_idx[0]].replace("_", " ").title(),
        "confidence":        round(float(predictions[top3_idx[0]]) * 100, 2),
        "top3_predictions":  top3
    }
