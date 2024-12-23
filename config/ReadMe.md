# Configuration

Training and model configurations:

## training_config.yaml
```yaml
task: detect
mode: train
model: yolov8l.pt
data: ../data/dataset.yaml
epochs: 150
batch: 16
imgsz: 640
patience: 50
```