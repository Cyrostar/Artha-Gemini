import os
import torch
import numpy as np

import base64
from io import BytesIO
from PIL import Image
import soundfile as sf

from google import genai
from google.genai import types

from ..core.node import node_path

def call_gemini_text_api(*args, **kwargs):
    
    text_prompt         = kwargs.get('text_prompt')
    image               = kwargs.get('image')
    system_instruction  = kwargs.get('system_instruction')
    api_key             = kwargs.get('api_key')
    model               = kwargs.get('model')
    max_tokens          = kwargs.get('max_tokens')
    temperature         = kwargs.get('temperature')
    seed                = kwargs.get('seed')
    top_k               = kwargs.get('top_k')
    top_p               = kwargs.get('top_p')
   
    client = genai.Client(api_key=api_key)
    
    generation_config = {}
    
    if temperature is not None:
        temp_value = max(0.0, min(2.0, float(temperature)))
        generation_config['temperature'] = temp_value
        
    if top_k is not None:
        generation_config['top_k'] = int(top_k)

    if top_p is not None:
        generation_config['top_p'] = float(top_p)
    
    if max_tokens is not None and int(max_tokens) > 0:
        generation_config['max_output_tokens'] = int(max_tokens)
        
    if seed is not None:
        generation_config['seed'] = normalize_seed(seed)
        
    if system_instruction and system_instruction.strip():
        full_prompt = f"System: {system_instruction.strip()}\n\nUser: {text_prompt}"
    else:
        full_prompt = text_prompt
        
    generation_config.update({
        'candidate_count': 1,
        'stop_sequences': [],
    })
    
    contents = []
    contents.append(full_prompt)
        
    if image is not None:
        contents.append(image)
        
    try:
        api_response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**generation_config)
        )
        
        text_parts = []
        
        for part in api_response.candidates[0].content.parts:
            if part.text is not None:
                text_parts.append(part.text)
        
        response = " ".join(text_parts) if text_parts else ""
        return response
        
    except Exception as e:
        
        print(f"API Error: {e}")
        return f"Error: {str(e)}"
        
def call_gemini_image_api(text_prompt, image, system_instruction, api_key, model, max_tokens, temperature):

    client = genai.Client(api_key=api_key)
    
    generation_config = {
        'response_modalities': ['TEXT', 'IMAGE']
    }
    
    if temperature is not None:
        
        temp_value = max(0.0, min(2.0, float(temperature)))
        generation_config['temperature'] = temp_value
    
    if max_tokens is not None and int(max_tokens) > 0:
        
        generation_config['max_output_tokens'] = int(max_tokens)
        
    if system_instruction and system_instruction.strip():
        
        full_prompt = f"System: {system_instruction.strip()}\n\nUser: {text_prompt}"
        
    else:
        
        full_prompt = text_prompt
        
    generation_config.update({
        'candidate_count': 1,
        'stop_sequences': [],
    })
    
    contents = []
    contents.append(full_prompt)
        
    if image is not None:
       
        contents.append(image)
        
    api_response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=contents,
        config=types.GenerateContentConfig(**generation_config)
    )
    
    text_parts = []
    image_tensor = None
    
    for part in api_response.candidates[0].content.parts:
        
        if part.text is not None:
            
            text_parts.append(part.text)          
            
        elif part.inline_data is not None:
            
            generated_image = Image.open(BytesIO((part.inline_data.data)))
            
            if generated_image.mode != 'RGB':
                
                generated_image = generated_image.convert('RGB')
            
            image_np = np.array(generated_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]  # Add batch dimension
        
    tensor = image_tensor if image_tensor is not None else None
    response = " ".join(text_parts) if text_parts else None
   
    return (tensor, response)
    
def call_gemini_tts_api(text_prompt, voice, api_key, temperature=None):
    
    client = genai.Client(api_key=api_key)
    
    generation_config = {
        'response_modalities': ['AUDIO'],  
        "speech_config": types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice,
                )
            )
        )
    }
    
    if temperature is not None:
        temp_value = max(0.0, min(2.0, float(temperature)))
        generation_config['temperature'] = temp_value
    
    try:
        
        api_response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=[text_prompt],
            config=types.GenerateContentConfig(**generation_config)
        )
        
        raw_audio_bytes = None
        
        for part in api_response.candidates[0].content.parts:
            
            if part.inline_data and part.inline_data.data:
                
                raw_audio_bytes = part.inline_data.data
                break
    
        if raw_audio_bytes is None:
            print("Error: Gemini API did not return audio data.")
            return None

  
        sample_rate = 24000
        audio_data, _ = sf.read(
            BytesIO(raw_audio_bytes), 
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

        return {"waveform": audio_tensor, "sample_rate": sample_rate}

    except Exception as e:
        
        print(f"!!! An exception occurred during Gemini TTS API call: {e}")
        return None

def gemini_api_parameters():
    
    api_parameters = {
        "api_key": ("STRING", {
            "multiline": False,
            "default": "",
            "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
        }),
        "model": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
            "default": "gemini-2.5-flash"
        }),
        "max_tokens": ("INT", {
            "default": 5000,
            "min": 1,
            "max": 8192,
            "step": 1,
            "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
        }),
        "temperature": ("FLOAT", {
            "default": 0.7,
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
            "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
        }),
    }
    
    return api_parameters
    
def load_agent(agent):
    
    node_dir = node_path()
    agent_path = os.path.join(node_dir, "nodes", "llm", "agents", agent + ".txt")
        
    try:
        
        with open(agent_path, 'r') as f:
            content = f.read()
            return content
        
    except Exception as e:
        
        print(f"Error loading text file: {e}")
        return None
        
def normalize_seed(seed):
    
    seed_max = 2147483647
    seed_int = abs(int(seed))
    
    return seed_int % (seed_max + 1)