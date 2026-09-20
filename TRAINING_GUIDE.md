# Training Guide: Eliminating Grid Artifacts

## Problem: Grid-like patterns in upscaled images

You're experiencing this because:

1. **Limited training data** - Only 64x64 cat images
2. **Single scale training** - Model only sees 32→64 upscaling
3. **Lack of diversity** - Only cats, no varied textures/scenes
4. **Tile artifacts** - Model learns patch-specific patterns

## Solution: Train on diverse, multi-scale data

### Step 1: Get Better Training Data

Download diverse, high-quality datasets:

#### Recommended Datasets:

1. **DIV2K** (800 high-quality 2K images)
   ```bash
   # Download from: https://data.vision.ee.ethz.ch/cvl/DIV2K/
   # Or use kaggle:
   # kaggle datasets download -d joe1995/div2k-dataset
   ```

2. **Flickr2K** (2650 high-quality images)
   ```bash
   # Download from: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
   ```

3. **COCO Dataset** (diverse scenes)
   ```bash
   # Download validation set (smaller):
   # https://cocodataset.org/#download
   ```

4. **General-100** (100 diverse images, good for testing)

### Step 2: Update the Training Script

Edit `train_improved_upscaler.py`:

```python
DATA_DIRS = [
    "/path/to/div2k/images",
    "/path/to/flickr2k/images", 
    "/path/to/coco/val2017",
    # Your cat dataset is fine to include too
    "/Users/laura/.cache/kagglehub/datasets/borhanitrash/cat-dataset/versions/1/cats/Data",
]
```

### Step 3: Run Improved Training

```bash
cd "/Users/laura/Tyler's Code/models/unblurrer"
source .venv/bin/activate
python train_improved_upscaler.py
```

This will:
- Train on multiple image scales (64, 128, 256)
- Use random crops from larger images
- Apply data augmentation (flips, rotations, brightness)
- Create 3x more training samples from the same images

### Step 4: Adjust for Better Results

#### For even less grid artifacts:

1. **Increase overlap in inference:**
   ```python
   # In use_flexible_upscaler.py, line ~193:
   upscaled = upscale_large_image_tiled(model, img, tile_size=512, overlap=128, ...)
   #                                                             ^^^^^^^ increase to 96 or 128
   ```

2. **Use larger training patches:**
   ```python
   TRAINING_SIZES = [128, 256, 512]  # Skip 64, use larger patches
   ```

3. **Train longer:**
   ```python
   EPOCHS = 150  # More epochs for better convergence
   ```

### Step 5: Alternative Quick Fix (Without Retraining)

If you can't retrain right now, try these inference tweaks:

1. **Increase tile overlap:**
   ```python
   # In use_flexible_upscaler.py, modify the tiled call:
   upscale_large_image_tiled(model, img, tile_size=384, overlap=128, ...)
   ```

2. **Use smaller tiles with more overlap:**
   ```python
   # Smaller tiles = more overlap relative to tile size
   upscale_large_image_tiled(model, img, tile_size=256, overlap=96, ...)
   ```

3. **Apply slight Gaussian blur to output:**
   ```python
   # After upscaling, very subtle blur to smooth grid:
   upscaled = cv2.GaussianBlur(upscaled, (3, 3), 0.5)
   ```

## Expected Improvements

With diverse training data:
- ✅ No grid patterns
- ✅ Better generalization to non-cat images
- ✅ Smoother textures
- ✅ Better edge preservation
- ✅ Works well on varied content (landscapes, text, people, etc.)

## Training Time Estimates

With the improved script:
- **M4 Mac (Metal GPU)**: ~2-4 hours for 100 epochs
- **With 5000+ training samples**: Best results
- **With data augmentation**: 3x more effective samples

## Quick Test After Training

```bash
# Test the improved model
python use_flexible_upscaler.py \
  -i test_image.jpg \
  -o upscaled_improved.jpg \
  -m improved_flexible_upscaler.keras \
  --scale 4
```

## Pro Tips

1. **Start with DIV2K** - It's specifically designed for super-resolution
2. **Mix dataset types** - Natural images + synthetic patterns work well together
3. **Monitor validation loss** - Should decrease steadily
4. **Save checkpoints** - Keep best model during training
5. **Test on diverse images** - Cats, landscapes, text, faces, etc.

## Files Created

- `train_improved_upscaler.py` - New training script with improvements
- This guide (`TRAINING_GUIDE.md`)

## Current Fix Applied

I've already fixed the tile blending artifacts in `use_flexible_upscaler.py`:
- ✅ Proper weighted blending with feathering
- ✅ Tiles upscaled completely before blending (fixes grey artifacts)
- ✅ Multi-pass upscaling uses tiled approach

The remaining grid pattern is from training data limitations.
