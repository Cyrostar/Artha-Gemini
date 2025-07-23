import os
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