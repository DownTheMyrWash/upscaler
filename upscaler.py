import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Optimize TensorFlow for performance
# tf.config.optimizer.set_jit(True)  # XLA disabled due to compatibility issues on macOS

# --- UPSCALER CONFIG ---
LOW_RES_SIZE = 32    # Input low resolution size (downsampled from your 64x64 images)
HIGH_RES_SIZE = 64   # Output high resolution size (your original 64x64 images)
SCALE_FACTOR = HIGH_RES_SIZE // LOW_RES_SIZE  # 2x upscaling
BATCH_SIZE = 32      # Good batch size for 64x64 output
EPOCHS = 75          # Optimal epochs for convergence
DATA_DIR = "./images"
NUM_WORKERS = multiprocessing.cpu_count()
MAX_IMAGES = 1500    # Fewer images due to memory constraints

print(f"=== IMAGE UPSCALER ===")
print(f"Upscaling from {LOW_RES_SIZE}x{LOW_RES_SIZE} to {HIGH_RES_SIZE}x{HIGH_RES_SIZE}")
print(f"Scale factor: {SCALE_FACTOR}x")

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
        
        # Create low resolution image (input) by downsampling
        low_res = cv2.resize(high_res, (low_size, low_size), interpolation=cv2.INTER_AREA)
        
        return low_res, high_res
    return None, None

def load_image_pairs(path, low_size, high_size):
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if MAX_IMAGES is not None and len(image_files) > MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
        print(f"Limited to {MAX_IMAGES} images for training")
    
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

# --- SPLIT DATA ---
x_train, x_test, y_train, y_test = train_test_split(
    low_res_images, high_res_images, test_size=0.2, random_state=42
)

print(f"Training data: {x_train.shape} -> {y_train.shape}")
print(f"Test data: {x_test.shape} -> {y_test.shape}")

# --- IMPROVED SUPER RESOLUTION MODEL ---
def build_enhanced_upscaler():
    """
    Build an enhanced Super Resolution model with attention mechanisms and skip connections
    """
    inputs = layers.Input(shape=(LOW_RES_SIZE, LOW_RES_SIZE, 3))
    
    # Initial feature extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Enhanced residual blocks with attention
    def enhanced_residual_block(x, filters, kernel_size=3):
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Channel attention mechanism
        gap = layers.GlobalAveragePooling2D()(x)
        gap = layers.Reshape((1, 1, filters))(gap)
        attention = layers.Conv2D(filters//8, (1, 1), activation='relu')(gap)
        attention = layers.Conv2D(filters, (1, 1), activation='sigmoid')(attention)
        x = layers.Multiply()([x, attention])
        
        # Skip connection
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x
    
    # Multiple enhanced residual blocks
    for i in range(8):  # More blocks for better feature learning
        x = enhanced_residual_block(x, 64)
    
    # Progressive upsampling with sub-pixel convolution
    # First upsampling stage
    x = layers.Conv2D(256, (3, 3), padding='same')(x)  # 64 * 4 channels for 2x upscale
    x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)  # 2x upscale using sub-pixel convolution
    x = layers.ReLU()(x)
    
    # Refinement layers after upsampling
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Final output layer
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Custom perceptual loss function
def perceptual_loss(y_true, y_pred):
    """
    Combine MSE loss with perceptual loss for better visual quality
    """
    # MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Edge-aware loss (emphasizes sharp edges)
    def sobel_edges(img):
        # Sobel edge detection
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Convert to grayscale
        gray_img = tf.reduce_mean(img, axis=-1, keepdims=True)
        
        edges_x = tf.nn.conv2d(gray_img, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        edges_y = tf.nn.conv2d(gray_img, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        edges = tf.sqrt(edges_x**2 + edges_y**2)
        return edges
    
    # Edge loss
    true_edges = sobel_edges(y_true)
    pred_edges = sobel_edges(y_pred)
    edge_loss = tf.reduce_mean(tf.square(true_edges - pred_edges))
    
    # Combine losses
    total_loss = mse_loss + 0.1 * edge_loss
    return total_loss

def build_upscaler():
    """
    Build the enhanced upscaler model
    """
    model = build_enhanced_upscaler()
    
    # Compile with perceptual loss and advanced optimizer
    model.compile(
        optimizer='adam',
        loss=perceptual_loss,
        metrics=['mae', 'mse']
    )
    
    return model

# Alternative ESPCN (Efficient Sub-Pixel CNN) model
def build_espcn_upscaler():
    """
    Build an ESPCN (Efficient Sub-Pixel Convolutional Neural Network) model
    """
    inputs = layers.Input(shape=(LOW_RES_SIZE, LOW_RES_SIZE, 3))
    
    # Feature extraction
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Sub-pixel convolution for upscaling
    # Calculate the number of channels needed for sub-pixel convolution
    channels_for_upscale = 3 * (SCALE_FACTOR ** 2)
    x = layers.Conv2D(channels_for_upscale, (3, 3), padding='same')(x)
    
    # Depth to space operation (sub-pixel shuffle)
    outputs = layers.Lambda(lambda x: tf.nn.depth_to_space(x, SCALE_FACTOR))(x)
    outputs = layers.Activation('sigmoid')(outputs)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# Choose which model to use
print("Building upscaler model...")
model = build_upscaler()  # Use SRCNN-style model
# model = build_espcn_upscaler()  # Alternative: use ESPCN model

model.summary()

# --- TRAIN ---
training_callbacks = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint('best_upscaler.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Starting upscaler training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=training_callbacks,
    verbose=1
)

# --- TEST VISUALIZATION ---
def show_upscaling_results(model, x_low, y_high, count=3):
    print("Generating upscaling predictions...")
    
    # Get predictions
    preds = model.predict(x_low[:count], verbose=0)
    preds = np.clip(preds, 0, 1)
    
    # Also create bicubic upscaling for comparison
    bicubic_upscaled = []
    for i in range(count):
        # Convert to uint8 for cv2.resize, then back to float
        img_uint8 = (x_low[i] * 255).astype(np.uint8)
        upscaled = cv2.resize(img_uint8, (HIGH_RES_SIZE, HIGH_RES_SIZE), interpolation=cv2.INTER_CUBIC)
        bicubic_upscaled.append(upscaled / 255.0)
    
    bicubic_upscaled = np.array(bicubic_upscaled)
    
    for i in range(count):
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        # Low resolution input
        axs[0, 0].imshow(x_low[i])
        axs[0, 0].set_title(f"Input ({LOW_RES_SIZE}x{LOW_RES_SIZE})")
        
        # Bicubic upscaling
        axs[0, 1].imshow(bicubic_upscaled[i])
        axs[0, 1].set_title(f"Bicubic Upscale ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})")
        
        # AI upscaling
        axs[1, 0].imshow(preds[i])
        axs[1, 0].set_title(f"AI Upscale ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})")
        
        # Ground truth high resolution
        axs[1, 1].imshow(y_high[i])
        axs[1, 1].set_title(f"Ground Truth ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})")
        
        for ax in axs.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'upscaler_result_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Upscaling results saved as upscaler_result_0.png to upscaler_result_{count-1}.png")
    
    # Calculate and print PSNR (Peak Signal-to-Noise Ratio)
    def calculate_psnr(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    ai_psnr = calculate_psnr(y_high[:count], preds)
    bicubic_psnr = calculate_psnr(y_high[:count], bicubic_upscaled)
    
    print(f"\nQuality Metrics:")
    print(f"AI Upscaler PSNR: {ai_psnr:.2f} dB")
    print(f"Bicubic Upscaler PSNR: {bicubic_psnr:.2f} dB")
    print(f"Improvement: {ai_psnr - bicubic_psnr:.2f} dB")

# Save the model
print("Saving upscaler model...")
model.save('image_upscaler.keras')
print("Model saved as 'image_upscaler.keras'")

# Generate test results
print("Generating upscaling test results...")
show_upscaling_results(model, x_test, y_test)

print("\n=== UPSCALER TRAINING COMPLETE ===")
print("Your image upscaler is ready!")
print(f"It can upscale {LOW_RES_SIZE}x{LOW_RES_SIZE} images to {HIGH_RES_SIZE}x{HIGH_RES_SIZE}")
