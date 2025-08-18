import torch
import numpy as np
from PIL import Image
import soundfile as sf
from io import BytesIO

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
        
def gemini_image_to_tensor(image):
    
    gemini_image = Image.open(BytesIO((image)))
            
    if gemini_image.mode != 'RGB':
        
        gemini_image = gemini_image.convert('RGB')
    
    image_np = np.array(gemini_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)[None,]
    
    return image_tensor
    
def gemini_tts_to_tensor(audio):
    
        sample_rate = 24000
        
        audio_data, _ = sf.read(
            BytesIO(audio), 
            samplerate=sample_rate, 
            channels=1, 
            format='RAW', 
            subtype='PCM_16'
        )
   
        audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
   
        max_val = torch.max(torch.abs(audio_tensor))
        
        if max_val > 0:
            
            audio_tensor = audio_tensor / max_val

        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        return audio_tensor, sample_rate
        
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