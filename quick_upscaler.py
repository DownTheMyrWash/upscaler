import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# --- QUICK UPSCALER TEST CONFIG ---
LOW_RES_SIZE = 32    # Smaller for faster training
HIGH_RES_SIZE = 64   # 2x upscaling
SCALE_FACTOR = HIGH_RES_SIZE // LOW_RES_SIZE
BATCH_SIZE = 32
EPOCHS = 15  # Fewer epochs for quick test
DATA_DIR = "./images"
NUM_WORKERS = multiprocessing.cpu_count()
MAX_IMAGES = 500  # Much fewer images for quick test

print(f"=== QUICK UPSCALER TEST ===")
print(f"Upscaling from {LOW_RES_SIZE}x{LOW_RES_SIZE} to {HIGH_RES_SIZE}x{HIGH_RES_SIZE}")

# --- DATA LOADING ---
def load_single_image_pair(args):
    fname, path, low_size, high_size = args
    img_path = os.path.join(path, fname)
    img = cv2.imread(img_path)
    if img is not None:
        # Load high resolution image (target)
        high_res = cv2.resize(img, (high_size, high_size))
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)
        high_res = high_res / 255.0
        
        # Create low resolution image (input)
        low_res = cv2.resize(high_res, (low_size, low_size), interpolation=cv2.INTER_AREA)
        
        return low_res, high_res
    return None, None

def load_image_pairs(path, low_size, high_size):
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if MAX_IMAGES is not None and len(image_files) > MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
        print(f"Limited to {MAX_IMAGES} images for quick test")
    
    print(f"Loading {len(image_files)} image pairs...")
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        args = [(fname, path, low_size, high_size) for fname in image_files]
        results = list(executor.map(load_single_image_pair, args))
    
    low_res_images = []
    high_res_images = []
    for low, high in results:
        if low is not None and high is not None:
            low_res_images.append(low)
            high_res_images.append(high)
    
    print(f"Successfully loaded {len(low_res_images)} image pairs")
    return np.array(low_res_images), np.array(high_res_images)

# Load image pairs
low_res_images, high_res_images = load_image_pairs(DATA_DIR, LOW_RES_SIZE, HIGH_RES_SIZE)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    low_res_images, high_res_images, test_size=0.2, random_state=42
)

print(f"Training: {x_train.shape} -> {y_train.shape}")

# --- SIMPLE UPSCALER MODEL ---
def build_simple_upscaler():
    inputs = layers.Input(shape=(LOW_RES_SIZE, LOW_RES_SIZE, 3))
    
    # Simple upscaling approach
    x = layers.UpSampling2D(size=(SCALE_FACTOR, SCALE_FACTOR), interpolation='bilinear')(inputs)
    
    # Feature refinement layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Output layer
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

model = build_simple_upscaler()
model.summary()

# --- TRAIN ---
callbacks_list = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

print("Starting quick upscaler training...")
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=callbacks_list,
    verbose=1
)

# --- TEST RESULTS ---
def show_results(model, x_low, y_high, count=3):
    preds = model.predict(x_low[:count], verbose=0)
    preds = np.clip(preds, 0, 1)
    
    # Bicubic comparison
    bicubic_upscaled = []
    for i in range(count):
        img_uint8 = (x_low[i] * 255).astype(np.uint8)
        upscaled = cv2.resize(img_uint8, (HIGH_RES_SIZE, HIGH_RES_SIZE), interpolation=cv2.INTER_CUBIC)
        bicubic_upscaled.append(upscaled / 255.0)
    
    for i in range(count):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        axs[0, 0].imshow(x_low[i])
        axs[0, 0].set_title(f"Input {LOW_RES_SIZE}x{LOW_RES_SIZE}")
        
        axs[0, 1].imshow(bicubic_upscaled[i])
        axs[0, 1].set_title(f"Bicubic {HIGH_RES_SIZE}x{HIGH_RES_SIZE}")
        
        axs[1, 0].imshow(preds[i])
        axs[1, 0].set_title(f"AI Upscale {HIGH_RES_SIZE}x{HIGH_RES_SIZE}")
        
        axs[1, 1].imshow(y_high[i])
        axs[1, 1].set_title(f"Ground Truth {HIGH_RES_SIZE}x{HIGH_RES_SIZE}")
        
        for ax in axs.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'quick_upscaler_{i}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Results saved as quick_upscaler_0.png to quick_upscaler_{count-1}.png")

# Save and test
model.save('quick_upscaler.keras')
print("Quick upscaler saved!")

show_results(model, x_test, y_test)
print("\n=== QUICK UPSCALER TEST COMPLETE ===")
