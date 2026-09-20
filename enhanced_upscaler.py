import tensorflow as tf
from keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# --- ENHANCED UPSCALER CONFIG ---
LOW_RES_SIZE = 32
HIGH_RES_SIZE = 64
SCALE_FACTOR = HIGH_RES_SIZE // LOW_RES_SIZE
BATCH_SIZE = 16
EPOCHS = 100
DATA_DIR = "./images"
NUM_WORKERS = multiprocessing.cpu_count()
MAX_IMAGES = 1500

print(f"=== ENHANCED UPSCALER V2 ===")
print(f"Upscaling from {LOW_RES_SIZE}x{LOW_RES_SIZE} to {HIGH_RES_SIZE}x{HIGH_RES_SIZE}")

# --- DATA LOADING ---
def load_single_image_pair(args):
    fname, path, low_size, high_size = args
    img_path = os.path.join(path, fname)
    img = cv2.imread(img_path)
    if img is not None:
        high_res = cv2.resize(img, (high_size, high_size))
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)
        high_res = high_res / 255.0
        
        # Random augmentation
        if np.random.random() > 0.5:
            high_res = np.fliplr(high_res)
        
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

# Load data
low_res_images, high_res_images = load_image_pairs(DATA_DIR, LOW_RES_SIZE, HIGH_RES_SIZE)
x_train, x_test, y_train, y_test = train_test_split(
    low_res_images, high_res_images, test_size=0.2, random_state=42
)

print(f"Training data: {x_train.shape} -> {y_train.shape}")

# --- ENHANCED MODEL ---
def enhanced_residual_block(x, filters, name_prefix):
    """Enhanced residual block with dilated convolutions and channel attention"""
    shortcut = x
    
    # First conv with dilation
    x = layers.Conv2D(filters, (3, 3), dilation_rate=1, activation='relu', padding='same', name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    
    # Second conv with different dilation
    x = layers.Conv2D(filters, (3, 3), dilation_rate=2, activation='relu', padding='same', name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    
    # Channel attention (Squeeze-and-Excitation)
    se = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
    se = layers.Dense(filters // 8, activation='relu', name=f'{name_prefix}_se1')(se)
    se = layers.Dense(filters, activation='sigmoid', name=f'{name_prefix}_se2')(se)
    se = layers.Reshape((1, 1, filters), name=f'{name_prefix}_reshape')(se)
    x = layers.Multiply(name=f'{name_prefix}_se_mul')([x, se])
    
    # Skip connection with channel matching
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same', name=f'{name_prefix}_skip')(shortcut)
    
    x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name_prefix}_relu')(x)
    return x

def build_enhanced_upscaler():
    """Enhanced upscaler with multi-scale features"""
    inputs = layers.Input(shape=(LOW_RES_SIZE, LOW_RES_SIZE, 3))
    
    # Initial feature extraction
    x = layers.Conv2D(64, (7, 7), activation='relu', padding='same', name='initial_conv')(inputs)
    
    # Multi-scale feature extraction
    # Scale 1: Original resolution
    scale1 = enhanced_residual_block(x, 64, 'scale1_block1')
    scale1 = enhanced_residual_block(scale1, 64, 'scale1_block2')
    
    # Scale 2: Half resolution for global context
    scale2 = layers.AveragePooling2D((2, 2), name='scale2_pool')(x)
    scale2 = enhanced_residual_block(scale2, 64, 'scale2_block1')
    scale2 = enhanced_residual_block(scale2, 64, 'scale2_block2')
    scale2 = layers.UpSampling2D((2, 2), interpolation='bilinear', name='scale2_up')(scale2)
    
    # Combine scales
    x = layers.Concatenate(name='scale_concat')([scale1, scale2])
    x = layers.Conv2D(64, (1, 1), activation='relu', padding='same', name='scale_fusion')(x)
    
    # More enhanced residual blocks
    for i in range(6):
        x = enhanced_residual_block(x, 64, f'enhanced_block_{i}')
    
    # Progressive upsampling
    x = layers.Conv2D(256, (3, 3), padding='same', name='upscale_prep')(x)
    x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2), name='pixel_shuffle')(x)
    x = layers.ReLU(name='upscale_relu')(x)
    
    # Post-upsampling refinement
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='refine1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='refine2')(x)
    
    # Output
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='output')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name='EnhancedUpscaler')

# Enhanced loss function
def enhanced_loss(y_true, y_pred):
    """Enhanced loss with L1, edge, and SSIM components"""
    # L1 loss (better for image quality than MSE)
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Edge loss
    def sobel_edges(img):
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        gray_img = tf.reduce_mean(img, axis=-1, keepdims=True)
        edges_x = tf.nn.conv2d(gray_img, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        edges_y = tf.nn.conv2d(gray_img, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        return tf.sqrt(edges_x**2 + edges_y**2)
    
    true_edges = sobel_edges(y_true)
    pred_edges = sobel_edges(y_pred)
    edge_loss = tf.reduce_mean(tf.abs(true_edges - pred_edges))
    
    # SSIM loss
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    # Combine losses
    total_loss = l1_loss + 0.1 * edge_loss + 0.2 * ssim_loss
    return total_loss

def build_upscaler():
    model = build_enhanced_upscaler()
    model.compile(
        optimizer='adam',
        loss=enhanced_loss,
        metrics=['mae', 'mse']
    )
    return model

# Build and train
print("Building enhanced upscaler...")
model = build_upscaler()
model.summary()

# Training callbacks
training_callbacks = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint('enhanced_best_upscaler.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Starting enhanced training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=training_callbacks,
    verbose='auto'
)

# Results visualization
def show_enhanced_results(model, x_low, y_high, count=3):
    print("Generating enhanced predictions...")
    preds = model.predict(x_low[:count], verbose=0)
    preds = np.clip(preds, 0, 1)
    
    # Bicubic for comparison
    bicubic_upscaled = []
    for i in range(count):
        img_uint8 = (x_low[i] * 255).astype(np.uint8)
        upscaled = cv2.resize(img_uint8, (HIGH_RES_SIZE, HIGH_RES_SIZE), interpolation=cv2.INTER_CUBIC)
        bicubic_upscaled.append(upscaled / 255.0)
    
    bicubic_upscaled = np.array(bicubic_upscaled)
    
    for i in range(count):
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        axs[0, 0].imshow(x_low[i])
        axs[0, 0].set_title(f"Input ({LOW_RES_SIZE}x{LOW_RES_SIZE})")
        
        axs[0, 1].imshow(bicubic_upscaled[i])
        axs[0, 1].set_title(f"Bicubic ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})")
        
        axs[1, 0].imshow(preds[i])
        axs[1, 0].set_title(f"Enhanced AI ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})")
        
        axs[1, 1].imshow(y_high[i])
        axs[1, 1].set_title(f"Ground Truth ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})")
        
        for ax in axs.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'enhanced_upscaler_result_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Enhanced results saved as enhanced_upscaler_result_0.png to enhanced_upscaler_result_{count-1}.png")
    
    # Calculate PSNR and SSIM
    def calculate_psnr(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    def calculate_ssim(y_true, y_pred):
        ssim_values = []
        for i in range(len(y_true)):
            ssim_val = tf.image.ssim(
                tf.expand_dims(y_true[i], 0), 
                tf.expand_dims(y_pred[i], 0), 
                max_val=1.0
            ).numpy()[0]
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    
    ai_psnr = calculate_psnr(y_high[:count], preds)
    ai_ssim = calculate_ssim(y_high[:count], preds)
    bicubic_psnr = calculate_psnr(y_high[:count], bicubic_upscaled)
    bicubic_ssim = calculate_ssim(y_high[:count], bicubic_upscaled)
    
    print(f"\n=== ENHANCED QUALITY METRICS ===")
    print(f"Enhanced AI Upscaler PSNR: {ai_psnr:.2f} dB")
    print(f"Enhanced AI Upscaler SSIM: {ai_ssim:.4f}")
    print(f"Bicubic Upscaler PSNR: {bicubic_psnr:.2f} dB")
    print(f"Bicubic Upscaler SSIM: {bicubic_ssim:.4f}")
    print(f"PSNR Improvement: +{ai_psnr - bicubic_psnr:.2f} dB")
    print(f"SSIM Improvement: +{ai_ssim - bicubic_ssim:.4f}")

# Save and test
model.save('enhanced_image_upscaler.keras')
print("Enhanced model saved!")

show_enhanced_results(model, x_test, y_test)
print("\n=== ENHANCED UPSCALER COMPLETE ===")
print("Your enhanced upscaler with multi-scale features is ready!")
