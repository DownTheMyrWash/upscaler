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

# --- CONFIG ---
IMG_SIZE = 64
BATCH_SIZE = 32  # Reduced batch size for better gradient stability
EPOCHS = 50  # More epochs for better convergence
DATA_DIR = "./images"  # Folder of sharp images
NUM_WORKERS = multiprocessing.cpu_count()
MAX_IMAGES = 2000  # More images for better training

# --- DATA LOADING ---
def load_single_image(args):
    fname, path, size = args
    img_path = os.path.join(path, fname)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img / 255.0
    return None

def load_images(path, size):
    # Get all valid image files
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limit number of images if specified
    if MAX_IMAGES is not None and len(image_files) > MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
        print(f"Limited to {MAX_IMAGES} images for faster training")
    
    print(f"Loading {len(image_files)} images...")
    
    # Use parallel processing for faster loading
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        args = [(fname, path, size) for fname in image_files]
        results = list(executor.map(load_single_image, args))
    
    # Filter out None results
    images = [img for img in results if img is not None]
    print(f"Successfully loaded {len(images)} images")
    return np.array(images)

sharp_images = load_images(DATA_DIR, IMG_SIZE)

# --- BLUR FUNCTION ---
def blur_single_image(args):
    img, blur_params = args
    # Random blur parameters for more diverse training data
    kernel_size = np.random.choice([3, 5, 7, 9])
    sigma = np.random.uniform(0.5, 2.0)
    
    # Make sure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def blur_images(images):
    print("Blurring images with random parameters...")
    # Use parallel processing for faster blurring
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        args = [(img, None) for img in images]  # blur_params not used but kept for compatibility
        blurred = list(executor.map(blur_single_image, args))
    return np.array(blurred)

blurred_images = blur_images(sharp_images)

# --- SPLIT ---
x_train, x_test, y_train, y_test = train_test_split(blurred_images, sharp_images, test_size=0.2)

# --- MODEL ---
def build_unblurrer():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Encoder with skip connections (U-Net style)
    # Block 1
    conv1_1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    conv1_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv1_1)
    pool1 = layers.MaxPooling2D((2,2))(conv1_2)
    
    # Block 2
    conv2_1 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
    conv2_2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv2_1)
    pool2 = layers.MaxPooling2D((2,2))(conv2_2)
    
    # Bottleneck
    conv3_1 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
    conv3_2 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(conv3_1)
    
    # Decoder with skip connections
    # Block 4
    up4 = layers.UpSampling2D((2,2))(conv3_2)
    concat4 = layers.Concatenate()([up4, conv2_2])  # Skip connection
    conv4_1 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(concat4)
    conv4_2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv4_1)
    
    # Block 5
    up5 = layers.UpSampling2D((2,2))(conv4_2)
    concat5 = layers.Concatenate()([up5, conv1_2])  # Skip connection
    conv5_1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(concat5)
    conv5_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv5_1)
    
    # Output layer
    outputs = layers.Conv2D(3, (1,1), activation='sigmoid', padding='same')(conv5_2)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use better loss function and optimizer
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

model = build_unblurrer()
model.summary()

# --- TRAIN ---
# Add callbacks for better training
training_callbacks = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# Create TensorFlow datasets for better performance
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Starting training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=training_callbacks,
    verbose=1
)

# --- TEST VISUALIZATION ---
def show_results(model, x, y, count=5):
    print("Generating predictions for visualization...")
    preds = model.predict(x[:count], verbose=0)
    
    for i in range(count):
        fig, axs = plt.subplots(1, 3, figsize=(12,4))
        axs[0].imshow(x[i])
        axs[0].set_title("Blurred")
        axs[1].imshow(preds[i])
        axs[1].set_title("Unblurred")
        axs[2].imshow(y[i])
        axs[2].set_title("Original")
        for ax in axs: ax.axis('off')
        plt.savefig(f'result_{i}.png', dpi=100, bbox_inches='tight')
        plt.close()  # Close to free memory
    
    print(f"Results saved as result_0.png to result_{count-1}.png")

# Save model before showing results
print("Saving model...")
model.save('unblur_model.keras')
print("Model saved as 'unblur_model.keras'")

print("Generating test results...")
show_results(model, x_test, y_test)
