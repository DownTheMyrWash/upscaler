import tensorflow as tf
from keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Custom layer to replace Lambda layers for proper serialization
class DepthToSpace(layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super(DepthToSpace, self).__init__(**kwargs)
        self.block_size = block_size
        
    def call(self, x):
        return tf.nn.depth_to_space(x, self.block_size)
    
    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        new_height = height * self.block_size if height else None
        new_width = width * self.block_size if width else None
        new_channels = channels // (self.block_size ** 2) if channels else None
        return (batch, new_height, new_width, new_channels)
    
    def get_config(self):
        config = super().get_config()
        config.update({'block_size': self.block_size})
        return config

# --- FLEXIBLE UPSCALER CONFIG ---
SCALE_FACTOR = 2     # 2x upscaling (can be changed to 3, 4, etc.)
BATCH_SIZE = 16      # Smaller batch for flexibility with different sizes
EPOCHS = 50
DATA_DIR = "/Users/laura/.cache/kagglehub/datasets/borhanitrash/cat-dataset/versions/1/cats/Data"
NUM_WORKERS = multiprocessing.cpu_count()
MAX_IMAGES = 2000

print(f"=== FLEXIBLE IMAGE UPSCALER ===")
print(f"Scale factor: {SCALE_FACTOR}x")
print("Works with any input size!")

# --- FLEXIBLE DATA LOADING ---
def load_flexible_image_pairs(path, scale_factor, target_sizes=None):
    """
    Load images at multiple sizes for flexible training
    """
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if MAX_IMAGES is not None and len(image_files) > MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
    
    print(f"Loading {len(image_files)} flexible image pairs...")
    
    # Define multiple training sizes if not specified
    if target_sizes is None:
        target_sizes = [64]  # Since source images are 64x64, use 32x32 -> 64x64 pairs
    
    low_res_images = []
    high_res_images = []
    
    for fname in image_files:
        img_path = os.path.join(path, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # For each target size, create a training pair
            for high_size in target_sizes:
                low_size = high_size // scale_factor
                
                # Create high resolution version
                high_res = cv2.resize(img, (high_size, high_size))
                high_res = high_res / 255.0
                
                # Create low resolution version by downsampling
                low_res = cv2.resize(high_res, (low_size, low_size), interpolation=cv2.INTER_AREA)
                
                low_res_images.append(low_res)
                high_res_images.append(high_res)
    
    print(f"Created {len(low_res_images)} training pairs with various sizes")
    return low_res_images, high_res_images

# --- FLEXIBLE MODEL ARCHITECTURE ---
def build_flexible_upscaler(scale_factor=2):
    """
    Build a fully convolutional upscaler that works with any input size
    """
    # Use None for height and width to accept any size
    inputs = layers.Input(shape=(None, None, 3), name='flexible_input')
    
    # Initial feature extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Enhanced residual blocks with attention
    def flexible_residual_block(x, filters, name_prefix):
        shortcut = x
        
        # Main convolution path
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.ReLU(name=f'{name_prefix}_relu1')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        
        # Channel attention mechanism
        gap = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
        gap = layers.Reshape((1, 1, filters), name=f'{name_prefix}_reshape')(gap)
        attention = layers.Conv2D(filters//8, (1, 1), activation='relu', name=f'{name_prefix}_att1')(gap)
        attention = layers.Conv2D(filters, (1, 1), activation='sigmoid', name=f'{name_prefix}_att2')(attention)
        x = layers.Multiply(name=f'{name_prefix}_multiply')([x, attention])
        
        # Skip connection
        x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name_prefix}_relu2')(x)
        return x
    
    # Multiple residual blocks
    for i in range(6):  # Reduced for efficiency
        x = flexible_residual_block(x, 64, f'res_block_{i}')
    
    # Upscaling layer - works with any input size
    if scale_factor == 2:
        # For 2x upscaling
        x = layers.Conv2D(256, (3, 3), padding='same')(x)  # 64 * 4 channels
        x = DepthToSpace(block_size=2)(x)
    elif scale_factor == 3:
        # For 3x upscaling
        x = layers.Conv2D(576, (3, 3), padding='same')(x)  # 64 * 9 channels
        x = DepthToSpace(block_size=3)(x)
    elif scale_factor == 4:
        # For 4x upscaling (two 2x stages)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = DepthToSpace(block_size=2)(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = DepthToSpace(block_size=2)(x)
    else:
        # For other scale factors, use UpSampling2D
        x = layers.UpSampling2D(size=(scale_factor, scale_factor), interpolation='bilinear')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.ReLU()(x)
    
    # Final refinement layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Output layer
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='FlexibleUpscaler')
    
    return model

# Custom loss function for flexible sizes with stability improvements
def flexible_perceptual_loss(y_true, y_pred):
    """
    Stable perceptual loss that works with different image sizes
    """
    # Clip predictions to prevent extreme values
    y_pred = tf.clip_by_value(y_pred, 0.001, 0.999)
    
    # MSE loss with stability check
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Check for NaN and replace with fallback
    mse_loss = tf.where(tf.math.is_nan(mse_loss), tf.constant(0.0), mse_loss)
    
    # Edge-aware loss using Sobel filters
    def sobel_edges(img):
        # Sobel kernels
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Convert to grayscale with stability
        gray_img = tf.reduce_mean(img, axis=-1, keepdims=True)
        gray_img = tf.clip_by_value(gray_img, 0.001, 0.999)
        
        edges_x = tf.nn.conv2d(gray_img, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        edges_y = tf.nn.conv2d(gray_img, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        edges = tf.sqrt(tf.maximum(edges_x**2 + edges_y**2, 1e-8))
        return edges
    
    # Edge loss with stability
    try:
        true_edges = sobel_edges(y_true)
        pred_edges = sobel_edges(y_pred)
        edge_loss = tf.reduce_mean(tf.square(true_edges - pred_edges))
        edge_loss = tf.where(tf.math.is_nan(edge_loss), tf.constant(0.1), edge_loss)
    except:
        edge_loss = tf.constant(0.1)
    
    total_loss = mse_loss + tf.multiply(0.05, edge_loss)  # Reduced edge weight for stability
    return tf.where(tf.math.is_nan(total_loss), tf.constant(1.0), total_loss)

# Build the flexible model
model = build_flexible_upscaler(scale_factor=SCALE_FACTOR)

# Use Adam with gradient clipping for stability
from keras.optimizers import Adam
optimizer = Adam(
    learning_rate=0.0001,  # Lower learning rate
    clipnorm=1.0  # Gradient clipping
)

model.compile(
    optimizer=optimizer,
    loss=flexible_perceptual_loss,
    metrics=['mae', 'mse']
)

model.summary()

# --- FLEXIBLE TRAINING ---
# Load training data with multiple sizes
low_res_list, high_res_list = load_flexible_image_pairs(DATA_DIR, SCALE_FACTOR)

# Convert to padded batches for training
def pad_to_same_size(images, target_size=None):
    """Pad images to the same size for batching"""
    if target_size is None:
        # Find the maximum dimensions
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        # Round up to nearest multiple of 32 for efficiency
        max_h = ((max_h + 31) // 32) * 32
        max_w = ((max_w + 31) // 32) * 32
    else:
        max_h, max_w = target_size, target_size
    
    padded_images = []
    for img in images:
        h, w = img.shape[:2]
        pad_h = max_h - h
        pad_w = max_w - w
        
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        padded_images.append(padded)
    
    return np.array(padded_images)

# --- FLEXIBLE INFERENCE FUNCTION ---
def upscale_any_size_image(model, input_image, scale_factor=2):
    """
    Upscale an image of any size using the trained model
    """
    # Ensure input is float32 and normalized
    if input_image.dtype != np.float32:
        input_image = input_image.astype(np.float32) / 255.0
    
    # Add batch dimension
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, axis=0)
    
    # Predict
    upscaled = model.predict(input_image, verbose=0)
    
    # Remove batch dimension and clip values
    upscaled = np.squeeze(upscaled)
    upscaled = np.clip(upscaled, 0, 1)
    
    return upscaled

# --- TEST WITH DIFFERENT SIZES ---
def test_flexible_upscaler(model, test_sizes=[32, 48, 64, 80]):
    """
    Test the upscaler with different input sizes
    """
    print("Testing flexible upscaler with different sizes...")
    
    # Create test images of different sizes
    for size in test_sizes:
        # Create a simple test pattern
        test_img = np.random.rand(size, size, 3).astype(np.float32)
        
        # Add some structure (checkerboard pattern)
        for i in range(size):
            for j in range(size):
                if (i // 8 + j // 8) % 2 == 0:
                    test_img[i, j] = [0.8, 0.2, 0.2]  # Red squares
                else:
                    test_img[i, j] = [0.2, 0.2, 0.8]  # Blue squares
        
        # Upscale
        upscaled = upscale_any_size_image(model, test_img, SCALE_FACTOR)
        
        print(f"Input size: {test_img.shape} -> Output size: {upscaled.shape}")
        
        # Save result
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.title(f"Input ({size}x{size})")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(upscaled)
        plt.title(f"Output ({upscaled.shape[0]}x{upscaled.shape[1]})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'flexible_test_{size}.png', dpi=100, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Pad images to same size for training
    print("Preparing training data...")
    x_train_padded = pad_to_same_size(low_res_list, target_size=32)  # 32x32 for low res
    y_train_padded = pad_to_same_size(high_res_list, target_size=64)  # 64x64 for high res

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_train_padded, y_train_padded, test_size=0.2, random_state=42
    )

    print(f"Training data: {x_train.shape} -> {y_train.shape}")

    # Training callbacks
    training_callbacks = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint('flexible_upscaler.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print("Starting flexible upscaler training...")
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=training_callbacks,
        verbose='auto'
    )

    # Save the flexible model
    print("Saving flexible upscaler...")
    model.save('flexible_upscaler.keras')

    # Test with different sizes
    test_flexible_upscaler(model)

    print("\n=== FLEXIBLE UPSCALER COMPLETE ===")
    print("Your model now works with ANY input size!")
    print("Use upscale_any_size_image() function to upscale images of any dimension.")

