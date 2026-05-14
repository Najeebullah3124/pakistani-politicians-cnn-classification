# Docker Deployment

## Build
```bash
cd app
docker build -t politician-classifier:latest .
```

## Run
```bash
docker run -d \
  --name politician-api \
  -p 8000:8000 \
  --restart unless-stopped \
  politician-classifier
```

## Test
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -F "file=@image.jpg"
```

## Image Details
- Base: python:3.10-slim
- Exposes: port 8000
- Framework: FastAPI + Uvicorn
- Model: ResNet50 (resnet_final.keras)