# DVC Dataset Versioning

## Setup
- DVC version: 3.x
- Remote: local Google Drive folder (`dvc_remote_storage/`)

## Tracked
- `dataset/` — 1,407 images across 16 classes
- Split: 1,050 train / 202 val / 155 test

## Commands Used
```bash
dvc init
dvc add dataset/
dvc remote add -d local_remote /path/to/dvc_remote_storage
dvc push
```

## Reproduce Dataset
```bash
git clone https://github.com/abdul-rehman-s/pakistani-politicians-cnn-classification
dvc pull
```

## Dataset Integrity
- Zero cross-split duplicates (verified via MD5 hashing)
- Zero internal duplicates per split
- All 16 classes have minimum 80 images