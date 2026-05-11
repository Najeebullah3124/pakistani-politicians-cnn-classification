# MLflow Experiment Tracking

## Setup
- Tracking URI: local file-based (`mlruns/`)
- Experiment name: `Pakistani_Politician_Classifier`

## Runs Logged
| Run |     Model      | Test Acc | Val Acc |       Params             |
|-----|----------------|----------|---------|--------------------------|
| 1   | EfficientNetB0 | 98.06%   | 99.01%  | LR=1e-4, Dropout=0.4/0.3 |
| 2   | ResNet50       | 99.35%   | 98.51%  | LR=1e-4, Dropout=0.4/0.3 |

## Parameters Tracked
- model_name, num_classes, image_size, batch_size
- phase1_lr, phase2_lr, frozen_layers
- dropout_1, dropout_2, dense_1, dense_2
- augmentation settings, train/val/test counts

## Metrics Tracked
- test_accuracy, weighted_precision, weighted_recall, weighted_f1
- best_val_accuracy, best_val_loss
- Per-epoch val_accuracy and val_loss (for charts)

## Artifacts Logged
- Confusion matrix heatmaps (both models)
- Training curves (both models)
- Classification reports (.txt)
- Full Keras model files