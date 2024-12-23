# Dataset

## Structure

```
data/
├── raw/            # Original videos/images
├── processed/      # Processed dataset
├── train/
├── val/
└── test/
```

## Dataset Preparation

1. Collect accident videos/images
2. Label data using [LabelImg](https://github.com/heartexlabs/labelImg)
3. Split into train/val/test sets
4. Update `dataset.yaml` with paths

## dataset.yaml format
```yaml
train: data/train/
val: data/val/
test: data/test/
nc: 1  # number of classes
names: ['accident']
```