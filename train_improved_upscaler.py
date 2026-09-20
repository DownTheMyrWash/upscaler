"""
Improved training script for the flexible upscaler with:
- Multiple datasets support
- Various training scales
- Data augmentation
- Better generalization
"""

import tensorflow as tf
import keras
from keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import random
from pathlib import Path

# Import the model architecture from flexible_upscaler.py
from flexible_upscaler import build_flexible_upscaler, flexible_perceptual_loss, DepthToSpace

# --- IMPROVED TRAINING CONFIG ---
SCALE_FACTOR = 2
BATCH_SIZE = 8  # Smaller for larger images
EPOCHS = 100
NUM_WORKERS = multiprocessing.cpu_count()

# Multiple patch sizes for training - each batch will have same-sized images
TRAINING_SIZES = [128]  # Train on multiple scales for better generalization

# Data directories - add your own!
DATA_DIRS = [
    str(Path(__file__).parent / "datasets" / "DIV2K_train_HR"),
]

if os.environ.get("UNBLURRER_DATA_DIRS"):
    DATA_DIRS = os.environ["UNBLURRER_DATA_DIRS"].split(os.pathsep)

MAX_IMAGES_PER_DIR = 800  # Limit per directory to balance (DIV2K has 800 images)

print(f"=== IMPROVED FLEXIBLE IMAGE UPSCALER ===")
print(f"Scale factor: {SCALE_FACTOR}x")
print(f"Training on multiple scales: {TRAINING_SIZES}")
print(f"Data directories: {len(DATA_DIRS)}")

# --- DATA AUGMENTATION ---
def augment_image(img):
    """Apply random augmentation to an image"""
    # Random horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random vertical flip
    if random.random() > 0.5:
        img = cv2.flip(img, 0)
    
    # Random rotation (90 degree increments)
    if random.random() > 0.5:
        k = random.randint(1, 3)  # 90, 180, or 270 degrees
        img = np.rot90(img, k)
    
    # Random brightness adjustment (subtle)
    if random.random() > 0.5:
        factor = random.uniform(0.9, 1.1)
        img = np.clip(img * factor, 0, 1)
    
    return img

# --- IMPROVED DATA LOADING ---
def load_diverse_image_pairs(data_dirs, scale_factor, training_sizes, max_per_dir=None):
    """
    Load images from multiple directories at multiple scales
    """
    all_low_res = []
    all_high_res = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: Directory not found: {data_dir}")
            continue
            
        print(f"\nLoading from: {data_dir}")
        
        image_files = [f for f in os.listdir(data_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if max_per_dir and len(image_files) > max_per_dir:
            random.shuffle(image_files)
            image_files = image_files[:max_per_dir]
        
        print(f"  Found {len(image_files)} images")
        
        for fname in tqdm(image_files, desc=f"  Processing"):
            img_path = os.path.join(data_dir, fname)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # For each training size, create multiple pairs with random crops
            for high_size in training_sizes:
                low_size = high_size // scale_factor
                
                # Try to extract random crops from larger images
                h, w = img.shape[:2]
                
                if h >= high_size and w >= high_size:
                    # Extract 2-3 random crops per size for larger images
                    num_crops = min(3, (h * w) // (high_size * high_size))
                    
                    for _ in range(num_crops):
                        # Random crop
                        y = random.randint(0, h - high_size)
                        x = random.randint(0, w - high_size)
                        crop = img[y:y+high_size, x:x+high_size]
                        
                        # Apply augmentation
                        crop = augment_image(crop.copy())
                        
                        # Normalize
                        high_res = crop / 255.0
                        
                        # Create low-res version
                        low_res = cv2.resize(high_res, (low_size, low_size), 
                                           interpolation=cv2.INTER_AREA)
                        
                        all_low_res.append(low_res)
                        all_high_res.append(high_res)
                else:
                    # Image too small, just resize
                    high_res = cv2.resize(img, (high_size, high_size))
                    high_res = high_res / 255.0
                    
                    # Apply augmentation
                    high_res = augment_image(high_res)
                    
                    low_res = cv2.resize(high_res, (low_size, low_size), 
                                       interpolation=cv2.INTER_AREA)
                    
                    all_low_res.append(low_res)
                    all_high_res.append(high_res)
    
    print(f"\nTotal training pairs created: {len(all_low_res)}")
    return all_low_res, all_high_res

# --- CUSTOM DATA GENERATOR FOR MIXED SIZES ---
class MixedSizeGenerator(keras.utils.Sequence):
    """Generator that yields batches of same-sized images"""
    
    def __init__(self, low_res_images, high_res_images, batch_size=8, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group images by size
        self.size_groups = {}
        for lr, hr in zip(low_res_images, high_res_images):
            size = lr.shape[0]
            if size not in self.size_groups:
                self.size_groups[size] = {'low': [], 'high': []}
            self.size_groups[size]['low'].append(lr)
            self.size_groups[size]['high'].append(hr)
        
        # Convert to numpy arrays
        for size in self.size_groups:
            self.size_groups[size]['low'] = np.array(self.size_groups[size]['low'])
            self.size_groups[size]['high'] = np.array(self.size_groups[size]['high'])
        
        self.sizes = list(self.size_groups.keys())
        self.on_epoch_end()
    
    def __len__(self):
        # Total number of batches across all sizes
        total_batches = 0
        for size in self.sizes:
            n_samples = len(self.size_groups[size]['low'])
            total_batches += int(np.ceil(n_samples / self.batch_size))
        return total_batches
    
    def on_epoch_end(self):
        # Shuffle indices for each size group
        if self.shuffle:
            for size in self.sizes:
                n_samples = len(self.size_groups[size]['low'])
                self.size_groups[size]['indices'] = np.random.permutation(n_samples)
        else:
            for size in self.sizes:
                n_samples = len(self.size_groups[size]['low'])
                self.size_groups[size]['indices'] = np.arange(n_samples)
    
    def __getitem__(self, index):
        # Distribute batches across size groups
        cumulative_batches = 0
        for size in self.sizes:
            n_samples = len(self.size_groups[size]['low'])
            n_batches = int(np.ceil(n_samples / self.batch_size))
            
            if index < cumulative_batches + n_batches:
                # This batch is from this size group
                local_idx = index - cumulative_batches
                start_idx = local_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                indices = self.size_groups[size]['indices'][start_idx:end_idx]
                
                X = self.size_groups[size]['low'][indices]
                y = self.size_groups[size]['high'][indices]
                
                return X, y
            
            cumulative_batches += n_batches
        
        # Fallback (shouldn't reach here)
        size = self.sizes[0]
        return self.size_groups[size]['low'][:self.batch_size], \
               self.size_groups[size]['high'][:self.batch_size]

if __name__ == "__main__":
    print("\n=== LOADING TRAINING DATA ===")
    
    # Load diverse training data
    low_res_images, high_res_images = load_diverse_image_pairs(
        DATA_DIRS, 
        SCALE_FACTOR, 
        TRAINING_SIZES,
        max_per_dir=MAX_IMAGES_PER_DIR
    )
    
    if len(low_res_images) == 0:
        print("ERROR: No training data loaded!")
        exit(1)
    
    print(f"\nTotal training samples: {len(low_res_images)}")
    
    # Split data
    train_lr, val_lr, train_hr, val_hr = train_test_split(
        low_res_images, high_res_images, test_size=0.1, random_state=42
    )
    
    print(f"Training samples: {len(train_lr)}")
    print(f"Validation samples: {len(val_lr)}")
    
    # Create generators
    train_gen = MixedSizeGenerator(train_lr, train_hr, batch_size=BATCH_SIZE)
    val_gen = MixedSizeGenerator(val_lr, val_hr, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n=== BUILDING MODEL ===")
    # Build model with flexible input - no fixed shape
    model = build_flexible_upscaler(scale_factor=SCALE_FACTOR)
    
    # IMPORTANT: Build the model with one of the training sizes to initialize weights
    # The model will still work with other sizes due to fully convolutional architecture
    dummy_size = TRAINING_SIZES[0] // SCALE_FACTOR  # Use smallest size for initialization
    model.build((None, dummy_size, dummy_size, 3))
    model.summary()
    
    # Compile with improved settings
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss=flexible_perceptual_loss,
        metrics=['mse', 'mae']
    )
    
    print(f"\n=== TRAINING ===")
    
    # Callbacks
    checkpoint_callback = callbacks.ModelCheckpoint(
        'improved_flexible_upscaler.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\n=== TRAINING COMPLETE ===")
    print("Model saved as: improved_flexible_upscaler.keras")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Mean Absolute Error')
    
    plt.tight_layout()
    plt.savefig('improved_training_history.png', dpi=150)
    print("Training history saved as: improved_training_history.png")
