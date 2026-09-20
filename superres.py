from diffusers import StableDiffusionUpscalePipeline
import torch
from PIL import Image
from io import BytesIO
import requests

device = "mps"
pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", 
    torch_dtype=torch.float16,  # Use half precision
    low_cpu_mem_usage=True
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()  # Reduces memory usage

url = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"  # replace with your image URL
resp = requests.get(url, stream=True)
resp.raise_for_status()
image = Image.open(BytesIO(resp.content)).convert("RGB")

# Run the upscaler (replace prompt with something if desired)
result = pipe(prompt="", image=image)
upscaled = result.images[0]
upscaled.save("upscaled.png")
print("Saved upscaled.png")