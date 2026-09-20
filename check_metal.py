#!/usr/bin/env python3

import tensorflow as tf
import sys

print("=== TensorFlow Metal Acceleration Check ===\n")

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check if Metal plugin is available
try:
    from tensorflow.python.platform import build_info
    from typing import cast 

    # cast to a plain dict to avoid typed stubs that restrict allowed keys
    info = cast(dict, build_info.build_info)
    print(f"Built with Metal: {info.get('is_metal_build', False)}")
except Exception:
    print("Could not determine Metal build status")

# List all physical devices
print("\n=== Available Devices ===")
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(f"  {device.device_type}: {device.name}")

# Check specifically for GPU devices
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"\nGPU devices found: {len(gpu_devices)}")
for i, gpu in enumerate(gpu_devices):
    print(f"  GPU {i}: {gpu}")

# Check logical devices
logical_devices = tf.config.list_logical_devices()
print(f"\nLogical devices: {len(logical_devices)}")
for device in logical_devices:
    print(f"  {device.device_type}: {device.name}")

# Test Metal performance
print("\n=== Performance Test ===")
with tf.device('/GPU:0' if gpu_devices else '/CPU:0'):
    # Create a simple computation
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    
    # Time the computation
    import time
    start = time.time()
    c = tf.matmul(a, b)
    tf.reduce_sum(c)  # Force execution
    end = time.time()
    
    device_name = '/GPU:0 (Metal)' if gpu_devices else '/CPU:0'
    print(f"Matrix multiplication on {device_name}: {end - start:.3f} seconds")

# Check memory info if available
if gpu_devices:
    print("\n=== GPU Memory Info ===")
    try:
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        current_mb = memory_info['current'] / (1024**2)
        peak_mb = memory_info['peak'] / (1024**2)
        print(f"Current GPU memory usage: {current_mb:.1f} MB")
        print(f"Peak GPU memory usage: {peak_mb:.1f} MB")
    except Exception as e:
        print(f"Could not get memory info: {e}")

print("\n=== Status Summary ===")
if gpu_devices:
    print("✅ Metal GPU acceleration is AVAILABLE and ACTIVE")
    print("   Your model training and inference should be accelerated!")
else:
    print("❌ No GPU devices found - using CPU only")
    print("   Consider installing tensorflow-metal for M1/M2/M3/M4 acceleration")
