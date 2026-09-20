"""
Use the improved flexible upscaler model to upscale an image
"""

import tensorflow as tf
import keras
from keras import layers
import numpy as np
import cv2
import sys
import os

# Import custom layers
from flexible_upscaler import DepthToSpace, flexible_perceptual_loss

def upscale_image(model, img, scale_factor=2):
    """
    Upscale an image using the flexible model
    Handles large images by processing in overlapping patches
    """
    h, w = img.shape[:2]
    
    # For smaller images, process directly
    if h <= 512 and w <= 512:
        img_normalized = img / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        upscaled = model.predict(img_input, verbose=0)[0]
        upscaled = np.clip(upscaled * 255.0, 0, 255).astype(np.uint8)
        return upscaled
    
    # For larger images, use tiled processing with overlap
    patch_size = 256
    overlap = 32
    stride = patch_size - overlap
    
    output_h = h * scale_factor
    output_w = w * scale_factor
    output = np.zeros((output_h, output_w, 3), dtype=np.float32)
    weight_map = np.zeros((output_h, output_w, 1), dtype=np.float32)
    
    # Create weight matrix for blending (to avoid seams)
    blend_weight = np.ones((patch_size * scale_factor, patch_size * scale_factor, 1))
    fade = overlap * scale_factor
    for i in range(fade):
        alpha = i / fade
        blend_weight[i, :] *= alpha
        blend_weight[-i-1, :] *= alpha
        blend_weight[:, i] *= alpha
        blend_weight[:, -i-1] *= alpha
    
    print(f"Processing large image ({h}x{w}) in patches...")
    
    num_patches = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract patch
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch = img[y:y_end, x:x_end]
            
            # Pad if necessary
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            
            # Process patch
            patch_normalized = patch / 255.0
            patch_input = np.expand_dims(patch_normalized, axis=0)
            upscaled_patch = model.predict(patch_input, verbose=0)[0]
            
            # Place in output with blending
            out_y = y * scale_factor
            out_x = x * scale_factor
            out_y_end = out_y + patch_size * scale_factor
            out_x_end = out_x + patch_size * scale_factor
            
            if out_y_end > output_h:
                out_y_end = output_h
            if out_x_end > output_w:
                out_x_end = output_w
            
            patch_h = out_y_end - out_y
            patch_w = out_x_end - out_x
            
            output[out_y:out_y_end, out_x:out_x_end] += upscaled_patch[:patch_h, :patch_w] * blend_weight[:patch_h, :patch_w]
            weight_map[out_y:out_y_end, out_x:out_x_end] += blend_weight[:patch_h, :patch_w]
            
            num_patches += 1
    
    print(f"Processed {num_patches} patches")
    
    # Normalize by weight map to get final result
    output = output / (weight_map + 1e-8)
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    
    return output

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Upscale an image using the improved flexible upscaler')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4, 8],
                        help='Upscaling factor (default: 2). Higher scales apply the 2x model multiple times.')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    target_scale = args.scale
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print("=== IMPROVED FLEXIBLE IMAGE UPSCALER ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target scale: {target_scale}x")
    
    # Load the model
    model_path = "improved_flexible_upscaler.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Please train the model first using train_improved_upscaler.py")
        sys.exit(1)
    
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'DepthToSpace': DepthToSpace,
            'flexible_perceptual_loss': flexible_perceptual_loss
        }
    )
    print("Model loaded successfully!")
    
    # Load input image
    print(f"\nLoading image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image: {input_path}")
        sys.exit(1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Input image size: {img.shape[1]}x{img.shape[0]}")
    
    # Calculate number of passes needed (model is 2x, so 4x needs 2 passes, 8x needs 3 passes)
    num_passes = target_scale // 2
    if target_scale not in [2, 4, 8]:
        print(f"Error: Scale must be 2, 4, or 8")
        sys.exit(1)
    
    # Upscale (apply 2x multiple times if needed)
    current_img = img
    for pass_num in range(num_passes):
        print(f"\nUpscaling pass {pass_num + 1}/{num_passes} (2x)...")
        current_img = upscale_image(model, current_img, scale_factor=2)
        print(f"Current size: {current_img.shape[1]}x{current_img.shape[0]}")
    
    upscaled = current_img
    print(f"\n✓ Final output image size: {upscaled.shape[1]}x{upscaled.shape[0]}")
    
    # Save result
    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, upscaled_bgr)
    print(f"\n✓ Upscaled image saved to: {output_path}")
    print(f"  Resolution increased from {img.shape[1]}x{img.shape[0]} to {upscaled.shape[1]}x{upscaled.shape[0]}")
