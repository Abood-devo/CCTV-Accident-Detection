# Accident Detection System

Real-time accident detection system using YOLOv8 and Gradio interface.

## Setup

1. Clone the repository:
```bash
git https://github.com/Abood-devo/CCTV-Accident-Detection.git
cd CCTV-Accident-Detection
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate accident-detection
```

3. Download pre-trained weights:
```bash
mkdir -p models/weights
# Download weights to models/weights/best.pt
```

## Usage

Start the Gradio interface:
```bash
python src/main.py
```

Access the interface at: http://localhost:7860

## Training

1. Prepare your dataset following instructions in `data/README.md`
2. Configure training parameters in `config/training_config.yaml`
3. Run training notebook: `notebooks/02_model_training.ipynb`

## Project Structure

- `src/`: Source code for inference and web interface
- `data/`: Dataset and data processing scripts 
- `models/`: Model weights and configurations
- `notebooks/`: Jupyter notebooks for training and evaluation
- `config/`: Configuration files
- `incidents/` : Detected accident clips and metadata
