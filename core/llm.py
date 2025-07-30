import os
import requests
import torch
import numpy as np
from io import BytesIO
from PIL import Image

import google.generativeai as genaix
from google import genai
from google.genai import types

from ..core.api import load_api_key
from ..core.node import node_path

def call_genai_api(text_prompt, image, system_instruction, api_key, model, max_tokens, temperature):

    # Initialize client with API key
    client = genai.Client(api_key=api_key)
    
    # Prepare comprehensive generation config
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
        model=model or "gemini-2.0-flash-preview-image-generation",
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
        
    image = image_tensor if image_tensor is not None else None
    response = " ".join(text_parts) if text_parts else None
    
    return (image, response)
               
 
def call_gemini_api(text_prompt, image, system_instruction, api_key, model, max_tokens, temperature):
    
    # Configure the API
    genaix.configure(api_key=api_key)
    
    # Create the model with system instruction
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    
    model_instance = genaix.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=system_instruction
    )
    
    content_parts = [text_prompt]
    
    if image is not None:
        
        if isinstance(image, list):
            
            for img in image:
                
                content_parts.append(img)
                    
        else:
            
            content_parts.append(image)     
    
    # Generate response
    response = model_instance.generate_content(content_parts)
    
    return response.text
    
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
        
        
def call_gemini_api_url(image_base64, text_prompt, system_instruction, api_key, model, max_tokens, temperature):
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    system_text_part = {"text": system_instruction}    
    system_instruction_object = {"parts": [system_text_part]}
    
    content_text_part = {"text": text_prompt} 
    content_image_part = {
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": image_base64
        }
    }  
    content_parts = [content_text_part, content_image_part]   
    content_object = {"parts": content_parts}   
    contents_array = [content_object]
      
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    
    payload = {
        "systemInstruction": system_instruction_object,
        "contents": contents_array,
        "generationConfig": generation_config
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the generated text
        if "candidates" in result and len(result["candidates"]) > 0:
            
            candidate = result["candidates"][0]
            
            if "content" in candidate and "parts" in candidate["content"]:
                
                return candidate["content"]["parts"][0]["text"]
            
            else:
                
                return "Error: No content in response"
        else:
            
            return f"Error: No candidates in response. Full response: {result}"
            
    except requests.exceptions.RequestException as e:
        
        return f"API Request Error: {str(e)}"
        
    except json.JSONDecodeError as e:
        
        return f"JSON Decode Error: {str(e)}"
        
    except Exception as e:
        
        return f"Unexpected Error: {str(e)}"