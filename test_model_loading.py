#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from keras.layers import Layer
import keras
import os

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

def test_model_loading():
    """Test if the model can be loaded successfully"""
    model_path = 'flexible_upscaler.keras'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found")
        print("ℹ️  You need to train the model first using flexible_upscaler.py")
        return False
    
    print(f"📁 Found model file: {model_path}")
    
    # Define custom objects
    custom_objects = {
        'DepthToSpace': DepthToSpace,
        'tf': tf,
        'depth_to_space': tf.nn.depth_to_space,
        'flexible_perceptual_loss': lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))
    }
    
    try:
        print("🔄 Loading model...")
        model = keras.models.load_model(
            model_path, 
            compile=False, 
            safe_mode=False,
            custom_objects=custom_objects
        )
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model summary:")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Total parameters: {model.count_params():,}")
        
        # Test with a small dummy input
        print("\n🧪 Testing model inference...")
        test_input = np.random.random((1, 32, 32, 3)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        print(f"✅ Inference successful!")
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    print("=== FLEXIBLE UPSCALER MODEL LOADING TEST ===\n")
    success = test_model_loading()
    print(f"\n{'='*50}")
    if success:
        print("🎉 All tests passed! The model can be loaded and used.")
    else:
        print("💥 Tests failed. Check the error messages above.")
