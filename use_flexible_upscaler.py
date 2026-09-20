import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import keras
from keras.layers import Layer
from tqdm import tqdm

# Enable unsafe deserialization for Lambda layers
keras.config.enable_unsafe_deserialization()

# Custom layer to replace problematic Lambda layers
class DepthToSpace(Layer):
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

def load_flexible_upscaler(model_path='flexible_upscaler.keras'):
    """Load the trained flexible upscaler model with custom objects"""
    custom_objects = {
        'DepthToSpace': DepthToSpace,
        'tf': tf,
        'depth_to_space': tf.nn.depth_to_space,
        'flexible_perceptual_loss': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))
    }
    
    try:
        return keras.models.load_model(
            model_path, 
            compile=False, 
            safe_mode=False,
            custom_objects=custom_objects
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: try loading without custom loss
        try:
            return keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            raise e2

def upscale_large_image_tiled(model, img, tile_size=512, overlap=64, scale_factor=2, verbose=True):
    """
    Upscale a large image using tiling to avoid OOM errors with proper blending.
    Each tile is upscaled to the full scale_factor before blending to avoid
    feeding the model its own output (which causes grey artifacts).
    """
    h, w, c = img.shape
    
    # Calculate how many 2x passes we need
    passes = 0
    current_scale = 1
    while current_scale < scale_factor:
        current_scale *= 2
        passes += 1
    final_scale = current_scale
    
    if verbose:
        print(f"Using tiled processing: {tile_size}x{tile_size} tiles with {overlap} pixel overlap")
        print(f"Each tile will be upscaled {final_scale}x ({passes} passes) before blending")
    
    # Process the entire image at once with proper tile blending
    output_h, output_w = h * final_scale, w * final_scale
    output_img = np.zeros((output_h, output_w, c), dtype=np.float32)
    weight_map = np.zeros((output_h, output_w, 1), dtype=np.float32)
    
    # Process image in overlapping tiles
    tile_count = 0
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            tile_count += 1
            
            # Extract tile with bounds checking
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img[y:y_end, x:x_end]
            
            # Upscale this tile through all passes before blending
            current_tile = tile
            for pass_num in range(passes):
                tile_batch = np.expand_dims(current_tile, axis=0)
                upscaled_tile_batch = model.predict(tile_batch, verbose=0)
                current_tile = np.squeeze(upscaled_tile_batch)
                current_tile = np.clip(current_tile, 0, 1)
            
            upscaled_tile = current_tile
            
            # Calculate output position
            out_y, out_x = y * final_scale, x * final_scale
            out_y_end, out_x_end = out_y + upscaled_tile.shape[0], out_x + upscaled_tile.shape[1]
            
            # Create feathering/blending weights for this tile
            tile_h, tile_w = upscaled_tile.shape[:2]
            tile_weight = np.ones((tile_h, tile_w, 1), dtype=np.float32)
            
            # Apply smooth feathering at tile edges if there's overlap
            if overlap > 0:
                overlap_scaled = overlap * final_scale  # Account for full upscaling
                
                # Feather the edges with a linear gradient
                for i in range(min(overlap_scaled, tile_h)):
                    # Top edge
                    if out_y > 0:
                        tile_weight[i, :, 0] *= (i + 1) / overlap_scaled
                    # Bottom edge
                    if out_y_end < output_h:
                        tile_weight[tile_h - 1 - i, :, 0] *= (i + 1) / overlap_scaled
                
                for j in range(min(overlap_scaled, tile_w)):
                    # Left edge
                    if out_x > 0:
                        tile_weight[:, j, 0] *= (j + 1) / overlap_scaled
                    # Right edge
                    if out_x_end < output_w:
                        tile_weight[:, tile_w - 1 - j, 0] *= (j + 1) / overlap_scaled
            
            # Accumulate weighted tile into output
            output_img[out_y:out_y_end, out_x:out_x_end] += upscaled_tile * tile_weight
            weight_map[out_y:out_y_end, out_x:out_x_end] += tile_weight
            
            if verbose and tile_count % 10 == 0:
                print(f"  Processed {tile_count} tiles...")
    
    if verbose:
        print(f"  Processed {tile_count} tiles total")
    
    # Normalize by accumulated weights to get final blended result
    output_img = np.divide(output_img, weight_map, where=weight_map > 0)
    
    # Handle any pixels that weren't covered (shouldn't happen but just in case)
    output_img = np.nan_to_num(output_img, nan=0.0)
    
    # Ensure final output is in valid range
    output_img = np.clip(output_img, 0, 1)
    
    return output_img

def upscale_image_file(model, input_path, output_path=None, scale_factor=2, verbose=True, max_size=2048):
    """
    Upscale an image file of any size
    
    Args:
        model: Trained upscaler model (2x upscaling)
        input_path: Path to input image
        output_path: Path to save upscaled image (optional)
        scale_factor: Upscaling factor (will apply 2x model iteratively if > 2)
        verbose: Whether to print progress information
        max_size: Maximum dimension before using tiled processing
    """
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        if verbose:
            print(f"Error: Could not load image from {input_path}")
        return None
    
    # Convert BGR to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    if verbose:
        print(f"Input image size: {img.shape}")
    
    # Calculate final scale
    passes = 0
    current_scale = 1
    while current_scale < scale_factor:
        current_scale *= 2
        passes += 1
    final_scale = current_scale
    
    if final_scale != scale_factor and verbose:
        print(f"Note: Scale factor {scale_factor} rounded to {final_scale} (nearest power of 2)")
    
    # Determine if we need tiled processing
    h, w = img.shape[:2]
    # Force tiled processing for multi-pass upscaling OR large images
    # This avoids the model seeing AI-upscaled images which it wasn't trained on
    if max(h, w) > max_size or passes > 1:
        if verbose:
            if passes > 1:
                print(f"Multi-pass upscaling detected. Using tiled processing for better quality.")
                print(f"(Each tile will be upscaled {final_scale}x in {passes} passes)")
            else:
                print(f"Large image detected ({h}x{w}). Using tiled processing to avoid memory issues.")
        upscaled = upscale_large_image_tiled(model, img, tile_size=384, overlap=96, 
                                           scale_factor=scale_factor, verbose=verbose)
    else:
        # Small enough for direct processing
        if verbose:
            print(f"Will apply {passes} passes of 2x upscaling for {final_scale}x total")
        
        current_img = img
        for i in range(passes):
            if verbose:
                print(f"Pass {i+1}/{passes}: {current_img.shape} -> ", end="")
                print(f"Input range: [{current_img.min():.3f}, {current_img.max():.3f}]")
            
            # Add batch dimension
            img_batch = np.expand_dims(current_img, axis=0)
            
            # Upscale
            upscaled_batch = model.predict(img_batch, verbose=0)
            
            # Remove batch dimension and clip
            current_img = np.squeeze(upscaled_batch)
            
            if verbose:
                print(f"Output range before clip: [{current_img.min():.3f}, {current_img.max():.3f}]")
            
            current_img = np.clip(current_img, 0, 1)
            
            if verbose:
                print(f"{current_img.shape}, range after clip: [{current_img.min():.3f}, {current_img.max():.3f}]")
        
        upscaled = current_img
    
    if verbose:
        print(f"Final output size: {upscaled.shape}")
    
    # Save if output path provided
    if output_path:
        # Convert back to BGR for saving with OpenCV
        upscaled_bgr = cv2.cvtColor((upscaled * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, upscaled_bgr)
        if verbose:
            print(f"Upscaled image saved to: {output_path}")
    
    return upscaled

def batch_upscale_directory(model, input_dir, output_dir, scale_factor=2, batch_size=64):
    """
    Upscale all images in a directory using true batching for efficiency
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    if not files:
        print("No supported image files found in the input directory.")
        return
    
    # Show info about the batch process
    total_files = len(files)
    print(f"Found {total_files} images to upscale with {scale_factor}x factor")
    print(f"Using batch size: {batch_size}")
    
    # Calculate passes needed for scale factor
    passes = 0
    current_scale = 1
    while current_scale < scale_factor:
        current_scale *= 2
        passes += 1
    
    if current_scale != scale_factor:
        print(f"Note: Scale factor {scale_factor} rounded to {current_scale}x (nearest power of 2)")
    
    # Process files in batches
    for i in tqdm(range(0, len(files), batch_size), desc="Processing batches", unit="batch"):
        batch_files = files[i:i + batch_size]
        
        # Load batch of images
        batch_images = []
        valid_files = []
        
        for filename in batch_files:
            input_path = os.path.join(input_dir, filename)
            img = cv2.imread(input_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                batch_images.append(img)
                valid_files.append(filename)
        
        if not batch_images:
            continue
            
        # Convert to numpy array
        batch_array = np.array(batch_images)
        
        # Apply multiple passes for higher scale factors
        current_batch = batch_array
        for pass_num in range(passes):
            # Process entire batch through model
            upscaled_batch = model.predict(current_batch, verbose=0)
            current_batch = np.clip(upscaled_batch, 0, 1)
        
        # Save upscaled images
        for j, filename in enumerate(valid_files):
            upscaled_img = current_batch[j]
            output_filename = f"upscaled_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Convert back to BGR for saving
            upscaled_bgr = cv2.cvtColor((upscaled_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, upscaled_bgr)
    
    print(f"Batch processing complete! Processed {total_files} images.")

def compare_upscaling_methods(input_image_path, model):
    """
    Compare your AI upscaler with traditional methods
    """
    # Load image
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    
    # Create low-res version for comparison
    low_res_size = (original_size[1] // 2, original_size[0] // 2)  # (width, height)
    low_res = cv2.resize(img, low_res_size, interpolation=cv2.INTER_AREA)
    
    # Traditional upscaling methods
    bicubic = cv2.resize(low_res, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
    lanczos = cv2.resize(low_res, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)
    
    # AI upscaling
    low_res_normalized = low_res.astype(np.float32) / 255.0
    ai_upscaled_batch = np.expand_dims(low_res_normalized, axis=0)
    ai_result = model.predict(ai_upscaled_batch, verbose=0)
    ai_upscaled = np.squeeze(ai_result)
    ai_upscaled = (np.clip(ai_upscaled, 0, 1) * 255).astype(np.uint8)
    
    # Display comparison
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(low_res)
    axs[0, 1].set_title('Low Resolution')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(bicubic)
    axs[0, 2].set_title('Bicubic Upscaling')
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(lanczos)
    axs[1, 0].set_title('Lanczos Upscaling')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(ai_upscaled)
    axs[1, 1].set_title('AI Upscaling')
    axs[1, 1].axis('off')
    
    # Calculate PSNR
    def calculate_psnr(img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    bicubic_psnr = calculate_psnr(img, bicubic)
    lanczos_psnr = calculate_psnr(img, lanczos)
    ai_psnr = calculate_psnr(img, ai_upscaled)
    
    # Show metrics
    metrics_text = f"PSNR Comparison:\nBicubic: {bicubic_psnr:.2f} dB\nLanczos: {lanczos_psnr:.2f} dB\nAI: {ai_psnr:.2f} dB"
    axs[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('upscaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flexible Image Upscaler')
    parser.add_argument('--input', '-i', required=True, help='Input image path or directory')
    parser.add_argument('--output', '-o', help='Output path or directory')
    parser.add_argument('--model', '-m', default='flexible_upscaler.keras', help='Path to model file')
    parser.add_argument('--scale', '-s', type=int, default=2, help='Scale factor')
    parser.add_argument('--batch', action='store_true', help='Force batch processing (for directories)')
    parser.add_argument('--compare', action='store_true', help='Compare with traditional methods')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading flexible upscaler model...")
    model = load_flexible_upscaler(args.model)
    print("Model loaded successfully!")
    
    # Determine if input is file or directory
    if os.path.isfile(args.input):
        # Single file processing
        if args.compare:
            # Compare methods for single file
            compare_upscaling_methods(args.input, model)
        else:
            # Regular upscaling for single file
            output_path = args.output or f"upscaled_{os.path.basename(args.input)}"
            print(f"Processing single file: {args.input}")
            upscale_image_file(model, args.input, output_path, args.scale)
            print(f"Single file processing complete!")
            
    elif os.path.isdir(args.input):
        # Directory processing (always use batch mode for directories)
        output_dir = args.output or f"{args.input.rstrip('/')}_upscaled"
        print(f"Processing directory: {args.input}")
        batch_upscale_directory(model, args.input, output_dir, args.scale)
        
    else:
        print(f"Error: Input path '{args.input}' is neither a valid file nor directory")
        exit(1)

# Example usage:
# python use_flexible_upscaler.py -i input_image.jpg -o output_image.jpg
# python use_flexible_upscaler.py -i input_folder/ -o output_folder/ --batch
# python use_flexible_upscaler.py -i test_image.jpg --compare
