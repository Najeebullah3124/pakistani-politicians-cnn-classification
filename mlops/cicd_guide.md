# GitHub Actions CI/CD Pipeline

## Workflow File
`.github/workflows/ci.yml`

## Pipeline Jobs

### Job 1: Test FastAPI App
- Sets up Python 3.10
- Installs lightweight dependencies (no TensorFlow)
- Runs 4 API tests:
  - Root endpoint
  - Health endpoint
  - Invalid file rejection
  - Valid image prediction

### Job 2: Build Docker Image
- Runs only if tests pass (`needs: test`)
- Builds the Docker image from `app/Dockerfile`
- Verifies image exists

## Triggers
- Every push to `main`
- Every pull request to `main`

## Status
![CI](https://github.com/abdul-rehman-s/pakistani-politicians-cnn-classification/actions/workflows/ci.yml/badge.svg)