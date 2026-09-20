import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# --- ULTRA ENHANCED UPSCALER CONFIG ---
LOW_RES_SIZE = 32    # Input low resolution size
HIGH_RES_SIZE = 64   # Output high resolution size (2x upscaling)
SCALE_FACTOR = HIGH_RES_SIZE // LOW_RES_SIZE
BATCH_SIZE = 16      # Smaller batch for more stable training
EPOCHS = 200         # More epochs for better convergence
DATA_DIR = "/Users/laura/.cache/kagglehub/datasets/borhanitrash/cat-dataset/versions/1/cats/Data"
NUM_WORKERS = multiprocessing.cpu_count()
MAX_IMAGES = 5000

print(f"=== ULTRA ENHANCED UPSCALER ===")
print(f"Upscaling from {LOW_RES_SIZE}x{LOW_RES_SIZE} to {HIGH_RES_SIZE}x{HIGH_RES_SIZE}")

# --- DATA LOADING WITH AUGMENTATION ---
def load_single_image_pair_augmented(args):
    fname, path, low_size, high_size = args
    img_path = os.path.join(path, fname)
    img = cv2.imread(img_path)
    if img is not None:
        # Load high resolution image (target)
        high_res = cv2.resize(img, (high_size, high_size))
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)
        high_res = high_res / 255.0
        
        # Random augmentation for better generalization
        if np.random.random() > 0.5:
            high_res = np.fliplr(high_res)  # Horizontal flip
        
        # Add slight color jitter
        if np.random.random() > 0.7:
            brightness = np.random.uniform(0.9, 1.1)
            high_res = np.clip(high_res * brightness, 0, 1)
        
        # Create low resolution image with slight blur for realism
        blur_kernel = np.random.choice([1, 3])
        if blur_kernel > 1:
            high_res_blurred = cv2.GaussianBlur(high_res, (blur_kernel, blur_kernel), 0.5)
        else:
            high_res_blurred = high_res
            
        low_res = cv2.resize(high_res_blurred, (low_size, low_size), interpolation=cv2.INTER_AREA)
        
        return low_res, high_res
    return None, None

def load_image_pairs(path, low_size, high_size):
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if MAX_IMAGES is not None and len(image_files) > MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
        print(f"Limited to {MAX_IMAGES} images for training")
    
    print(f"Loading {len(image_files)} image pairs with augmentation...")
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        args = [(fname, path, low_size, high_size) for fname in image_files]
        results = list(executor.map(load_single_image_pair_augmented, args))
    
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

# Ensure consistent data types (float32)
low_res_images = low_res_images.astype(np.float32)
high_res_images = high_res_images.astype(np.float32)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    low_res_images, high_res_images, test_size=0.2, random_state=42
)

print(f"Training data: {x_train.shape} -> {y_train.shape}")
print(f"Test data: {x_test.shape} -> {y_test.shape}")

# --- ULTRA ENHANCED MODEL ARCHITECTURE ---
class SelfAttention(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.query_conv = layers.Conv2D(channels // 8, 1, use_bias=False)
        self.key_conv = layers.Conv2D(channels // 8, 1, use_bias=False)
        self.value_conv = layers.Conv2D(channels, 1, use_bias=False)
        self.gamma = self.add_weight(name="gamma", shape=(), initializer="zeros", trainable=True)
        
    def call(self, x):
        batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Generate query, key, value
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # Reshape for attention computation
        query = tf.reshape(query, [batch_size, height * width, channels // 8])
        key = tf.reshape(key, [batch_size, height * width, channels // 8])
        value = tf.reshape(value, [batch_size, height * width, channels])
        
        # Compute attention
        attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True), axis=-1)
        out = tf.matmul(attention, value)
        out = tf.reshape(out, [batch_size, height, width, channels])
        
        # Apply gamma scaling and residual connection
        return self.gamma * out + x

def dense_block(x, growth_rate, num_layers, name_prefix):
    """Dense block with growth rate for feature reuse"""
    concat_list = [x]
    
    for i in range(num_layers):
        # Concatenate all previous layers
        concat_x = layers.Concatenate()(concat_list) if len(concat_list) > 1 else x
        
        # 1x1 conv to reduce channels
        bn1 = layers.BatchNormalization()(concat_x)
        relu1 = layers.ReLU()(bn1)
        conv1 = layers.Conv2D(growth_rate * 4, (1, 1), padding='same', name=f'{name_prefix}_conv1x1_{i}')(relu1)
        
        # 3x3 conv for feature extraction
        bn2 = layers.BatchNormalization()(conv1)
        relu2 = layers.ReLU()(bn2)
        conv2 = layers.Conv2D(growth_rate, (3, 3), padding='same', name=f'{name_prefix}_conv3x3_{i}')(relu2)
        
        concat_list.append(conv2)
    
    return layers.Concatenate()(concat_list)

def build_ultra_enhanced_upscaler():
    """
    Ultra Enhanced Super Resolution model with:
    - Dense blocks for feature reuse
    - Self-attention for long-range dependencies
    - Progressive upsampling
    - Skip connections
    """
    inputs = layers.Input(shape=(LOW_RES_SIZE, LOW_RES_SIZE, 3))
    
    # Initial feature extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='initial_conv')(inputs)
    
    # Store for skip connection
    skip_connection = x
    
    # Dense blocks for feature extraction
    x = dense_block(x, growth_rate=32, num_layers=4, name_prefix='dense1')
    x = layers.Conv2D(64, (1, 1), padding='same', name='transition1')(x)  # Reduce channels
    
    x = dense_block(x, growth_rate=32, num_layers=4, name_prefix='dense2')
    x = layers.Conv2D(64, (1, 1), padding='same', name='transition2')(x)
    
    # Self-attention for global context
    x = SelfAttention(64, name='self_attention')(x)
    
    # More dense blocks
    x = dense_block(x, growth_rate=32, num_layers=4, name_prefix='dense3')
    x = layers.Conv2D(64, (1, 1), padding='same', name='transition3')(x)
    
    # Add skip connection
    x = layers.Add()([x, skip_connection])
    
    # Enhanced residual blocks with squeeze-and-excitation
    def se_residual_block(x, filters, name_prefix):
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.ReLU(name=f'{name_prefix}_relu1')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        
        # Squeeze-and-Excitation
        se = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
        se = layers.Dense(filters // 4, activation='relu', name=f'{name_prefix}_se1')(se)
        se = layers.Dense(filters, activation='sigmoid', name=f'{name_prefix}_se2')(se)
        se = layers.Reshape((1, 1, filters), name=f'{name_prefix}_reshape')(se)
        x = layers.Multiply(name=f'{name_prefix}_se_mul')([x, se])
        
        # Skip connection
        x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name_prefix}_relu_out')(x)
        return x
    
    # Multiple SE residual blocks
    for i in range(6):
        x = se_residual_block(x, 64, f'se_res_{i}')
    
    # Progressive upsampling with pixel shuffle
    # First stage: 32x32 -> 64x64
    x = layers.Conv2D(256, (3, 3), padding='same', name='upscale_conv')(x)  # 64 * 4 for 2x upscale
    x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2), name='pixel_shuffle')(x)
    x = layers.ReLU(name='upscale_relu')(x)
    
    # Refinement after upsampling
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='refine1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='refine2')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='refine3')(x)
    
    # Final output layer
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='UltraEnhancedUpscaler')
    return model

# Advanced loss function with multiple components
def advanced_perceptual_loss(y_true, y_pred):
    """
    Advanced loss combining:
    - L1 loss (better than MSE for image quality)
    - Edge loss (Sobel)
    - SSIM loss (structural similarity)
    - Total variation loss (smoothness)
    """
    # L1 loss
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Edge loss using Sobel
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
    
    # Total variation loss for smoothness
    tv_loss = tf.reduce_mean(tf.image.total_variation(y_pred))
    
    # Combine losses
    total_loss = l1_loss + 0.1 * edge_loss + 0.1 * ssim_loss + 0.001 * tv_loss
    return total_loss

def build_upscaler():
    model = build_ultra_enhanced_upscaler()
    
    # Compile with advanced optimizer
    model.compile(
        optimizer='adam',
        loss=advanced_perceptual_loss,
        metrics=['mae', 'mse']
    )
    
    return model

# Build model
print("Building ultra enhanced upscaler...")
model = build_upscaler()
model.summary()

# --- ADVANCED TRAINING ---
# Advanced callbacks
training_callbacks = [
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=8, 
        min_lr=1e-8, 
        verbose=1,
        cooldown=2
    ),
    callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True, 
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        'ultra_best_upscaler.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    ),
    callbacks.LearningRateScheduler(
        lambda epoch: 0.0001 * (0.95 ** epoch),
        verbose=1
    )
]

# Create datasets with better preprocessing
def preprocess_data(x, y):
    # Ensure consistent data types
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    
    # Add slight noise for robustness
    noise = tf.random.normal(tf.shape(x), stddev=0.01, dtype=tf.float32)
    x_noisy = tf.clip_by_value(x + noise, 0.0, 1.0)
    return x_noisy, y

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Starting ultra enhanced training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=training_callbacks,
    verbose=1
)

# --- ENHANCED VISUALIZATION ---
def show_ultra_results(model, x_low, y_high, count=3):
    print("Generating ultra enhanced predictions...")
    
    preds = model.predict(x_low[:count], verbose=0)
    preds = np.clip(preds, 0, 1)
    
    # Bicubic comparison
    bicubic_upscaled = []
    for i in range(count):
        img_uint8 = (x_low[i] * 255).astype(np.uint8)
        upscaled = cv2.resize(img_uint8, (HIGH_RES_SIZE, HIGH_RES_SIZE), interpolation=cv2.INTER_CUBIC)
        bicubic_upscaled.append(upscaled / 255.0)
    
    bicubic_upscaled = np.array(bicubic_upscaled)
    
    for i in range(count):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        axs[0, 0].imshow(x_low[i])
        axs[0, 0].set_title(f"Input ({LOW_RES_SIZE}x{LOW_RES_SIZE})", fontsize=14)
        
        axs[0, 1].imshow(bicubic_upscaled[i])
        axs[0, 1].set_title(f"Bicubic ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})", fontsize=14)
        
        axs[1, 0].imshow(preds[i])
        axs[1, 0].set_title(f"Ultra AI Upscale ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})", fontsize=14)
        
        axs[1, 1].imshow(y_high[i])
        axs[1, 1].set_title(f"Ground Truth ({HIGH_RES_SIZE}x{HIGH_RES_SIZE})", fontsize=14)
        
        for ax in axs.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'ultra_upscaler_result_{i}.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"Ultra results saved as ultra_upscaler_result_0.png to ultra_upscaler_result_{count-1}.png")
    
    # Enhanced quality metrics
    def calculate_metrics(y_true, y_pred):
        # PSNR
        mse = np.mean((y_true - y_pred) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # SSIM
        ssim_values = []
        for i in range(len(y_true)):
            ssim_val = tf.image.ssim(
                tf.expand_dims(y_true[i], 0), 
                tf.expand_dims(y_pred[i], 0), 
                max_val=1.0
            ).numpy()[0]
            ssim_values.append(ssim_val)
        
        avg_ssim = np.mean(ssim_values)
        
        return psnr, avg_ssim
    
    ai_psnr, ai_ssim = calculate_metrics(y_high[:count], preds)
    bicubic_psnr, bicubic_ssim = calculate_metrics(y_high[:count], bicubic_upscaled)
    
    print(f"\n=== ULTRA ENHANCED QUALITY METRICS ===")
    print(f"Ultra AI Upscaler:")
    print(f"  PSNR: {ai_psnr:.2f} dB")
    print(f"  SSIM: {ai_ssim:.4f}")
    print(f"Bicubic Upscaler:")
    print(f"  PSNR: {bicubic_psnr:.2f} dB") 
    print(f"  SSIM: {bicubic_ssim:.4f}")
    print(f"Improvements:")
    print(f"  PSNR: +{ai_psnr - bicubic_psnr:.2f} dB")
    print(f"  SSIM: +{ai_ssim - bicubic_ssim:.4f}")

# Save the model
print("Saving ultra enhanced upscaler...")
model.save('ultra_image_upscaler.keras')
print("Model saved as 'ultra_image_upscaler.keras'")

# Generate results
print("Generating ultra enhanced results...")
show_ultra_results(model, x_test, y_test)

print("\n=== ULTRA ENHANCEMENT COMPLETE ===")
print("Your ultra enhanced upscaler is ready with cutting-edge features!")
