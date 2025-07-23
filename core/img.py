import torch
import numpy as np
from PIL import Image
import io
import base64

def tensor_to_base64(tensor):

    # Take first image from batch if needed
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    # Convert to numpy
    numpy_image = tensor.cpu().numpy()
    
    # Ensure values are in 0-255 range
    if numpy_image.max() <= 1.0:
        numpy_image = (numpy_image * 255).astype(np.uint8)
    else:
        numpy_image = numpy_image.astype(np.uint8)
    
    # Create PIL Image (still needed for JPEG encoding)
    pil_image = Image.fromarray(numpy_image)
    
    # Convert to base64
    buffered = io.BytesIO()
    
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    pil_image.save(buffered, format="JPEG")
    
    img = base64.b64encode(buffered.getvalue()).decode()
    
    return img
    
    
def tensor_to_pil_image(tensor):
 
    # Handle batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert from torch tensor to numpy array
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Ensure values are in 0-255 range
    if tensor.dtype == np.float32 or tensor.dtype == np.float64:
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).astype(np.uint8)
        else:
            tensor = tensor.astype(np.uint8)
    
    # Convert to PIL Image
    if tensor.shape[-1] == 3:  # RGB
        return Image.fromarray(tensor, 'RGB')
    elif tensor.shape[-1] == 4:  # RGBA
        return Image.fromarray(tensor, 'RGBA')
    else:
        # Handle grayscale or other formats
        return Image.fromarray(tensor.squeeze(), 'L')
        
def resize_image_shortest(image,size):
    
    width, height = image.size
    
    if width < height:

        new_width = size
        new_height = int(height * (size / width))
        
    else:

        new_height = size
        new_width = int(width * (size / height))
    
    image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image_resized