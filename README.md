# Image Unblurrer and Upscaler

TensorFlow/Keras experiments for removing blur and increasing image resolution. The repository contains several model architectures, training scripts, and inference utilities for 2x and multi-pass upscaling.

## Project Layout

- `train.py`: trains the original 64x64 image deblurring model.
- `train_improved_upscaler.py`: trains the flexible upscaler on one or more image datasets.
- `flexible_upscaler.py`, `enhanced_upscaler.py`, `ultra_upscaler.py`: model architectures and training variants.
- `use_flexible_upscaler.py`, `use_improved_upscaler.py`: inference utilities.
- `test_model_loading.py`: loads `flexible_upscaler.keras` and performs a small inference check.
- `TRAINING_GUIDE.md`: notes on datasets, training, and reducing tiling artifacts.

Datasets, trained weights, generated images, logs, and experiment tracking files are intentionally excluded from Git. Keep them locally or store them in an external artifact or model registry.

## Setup

PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands from `cmd.exe` with `.venv\Scripts\activate.bat`, or configure the execution policy for your user account according to your local Windows policy.

## Training

Place training images in a local dataset directory. For the improved trainer, set `UNBLURRER_DATA_DIRS` to a semicolon-separated list of directories:

```powershell
$env:UNBLURRER_DATA_DIRS = "datasets\DIV2K_train_HR"
python train_improved_upscaler.py
```

The original trainer expects images in `images\`:

```powershell
python train.py
```

Training writes model weights, logs, and visualizations into the project directory. Those files are ignored by Git.

## Inference

After training or downloading a compatible local model, run the flexible upscaler with an input image:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python use_flexible_upscaler.py `
  -i path\to\input.jpg `
  -o outputs\upscaled.jpg `
  -m flexible_upscaler.keras `
  --scale 4
```

The exact options available can be checked with:

```powershell
python use_flexible_upscaler.py --help
```

## Testing

The model-loading check requires the ignored `flexible_upscaler.keras` file:

```powershell
python test_model_loading.py
```

## Initialize Git Locally

Run these commands from this project folder:

```powershell
git init
git add .
git status
git commit -m "Initial commit"
git branch -M main
```

Review `git status` before committing to confirm that datasets, model weights, generated media, logs, and local environments are not staged.

## Connect GitHub and Push

Create an empty repository on GitHub first. Do not add a README, `.gitignore`, or license there because this folder already contains those files. Then replace `YOUR-USERNAME` and `YOUR-REPOSITORY` below:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git remote -v
git push -u origin main
```

For SSH instead:

```powershell
git remote add origin git@github.com:YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

To check or change the remote later:

```powershell
git remote -v
git remote set-url origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```
