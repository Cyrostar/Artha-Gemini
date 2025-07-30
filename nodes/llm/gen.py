import os

from PIL import Image

import folder_paths

from ...core.node import node_path, node_prefix, main_cetegory
from ...core.api import load_api_key
from ...core.llm import call_genai_api

class GeminiScenery:
    
    CATEGORY = main_cetegory() + "/LLM"
    DESCRIPTION = "Gemini image generation and modification."
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cat with a hat"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash-preview-image-generation"], {
                    "default": "gemini-2.0-flash-preview-image-generation"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
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
                "modify_image": ("BOOLEAN", {"default": False}),
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "system_instruction": ("ARTHAINSTRUCT", {
                    "forceInput": True,
                    "multiline": True
                }),                
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
           
    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("image","response",)
    FUNCTION = "artha_main"

    def artha_main(self, text_prompt, image, modify_image, api_key, model, max_tokens, temperature, system_instruction="", extra_pnginfo=None):
        
        tensor = None
        response = None
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (tensor, response,)
        
        pil_image = None
        
        if modify_image:
        
            image_path = folder_paths.get_annotated_filepath(image)
        
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (tensor, response,)
            
        try:
                              
            # Call Gemini API
            tensor, response = call_genai_api(
                text_prompt,
                pil_image,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
        
            return (tensor, response)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (tensor, response,)
                             
         
#################################################        

# Required mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Gemini Scenery": GeminiScenery,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini Scenery": node_prefix() + " Gemini Scenery",

}