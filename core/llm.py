import os
import requests
import google.generativeai as genai
from ..core.api import load_api_key
from ..core.node import node_path
    
def call_gemini_api(text_prompt, image, system_instruction, api_key, model, max_tokens, temperature):
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Create the model with system instruction
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    
    model_instance = genai.GenerativeModel(
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