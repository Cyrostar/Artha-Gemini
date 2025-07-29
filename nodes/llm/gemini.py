import os

from ...core.node import node_prefix, main_cetegory
from ...core.api import load_api_key
from ...core.img import tensor_to_pil_image, resize_image_shortest
from ...core.llm import call_gemini_api
from ...core.llm import load_agent

################################################# 

class GeminiQuestion:
    
    CATEGORY = main_cetegory() + "/LLM"
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
                    "default": "gemini-2.5-flash"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),              
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, question, api_key, model, max_tokens, temperature=0.7):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
            
        system_instruction = "You are an intelligent ai asistant."
            
        text_prompt = question
        
        try:
           
            image_base64 = None
            
            # Call Gemini API
            response = call_gemini_api(
                text_prompt,
                image_base64,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (error_msg,)
            
################################################# 

class GeminiOperation:
    
    CATEGORY = main_cetegory() + "/LLM"
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("STRING", {
                    "multiline": True,
                    "default": "A cat with a hat"
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "Change cat to dog"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
                    "default": "gemini-2.5-flash"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),              
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, source, instruction, api_key, model, max_tokens, temperature=0.7):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
            
        system_instruction = "Role: You are an intelligent ai asistant. \n\n"
        system_instruction += "Task: You will perform the given action on the text prompt.\n\n"
        system_instruction += "Action: " + instruction
            
        text_prompt = source
        
        try:
           
            image_base64 = None
            
            # Call Gemini API
            response = call_gemini_api(
                text_prompt,
                image_base64,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (error_msg,)
            
################################################# 

class GeminiTranslate:
    
    CATEGORY = main_cetegory() + "/LLM"
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "lang_from": (["Chinese", "Spanish", "English", "Hindi", "Portuguese", "Bengali", "Russian", "Japanese", "Turkish", "German", "French", "Italian"], {
                    "default": "Chinese"
                }),
                "lang_to": (["Chinese", "Spanish", "English", "Hindi", "Portuguese", "Bengali", "Russian", "Japanese", "Turkish", "German", "French", "Italian"], {
                    "default": "English"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
                    "default": "gemini-2.5-flash"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),               
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, text, lang_from, lang_to, api_key, model, max_tokens, temperature=0.7):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
                  
        system_instruction = "Translate the given text from " + lang_from + " to " + lang_to + "."
        system_instruction = "Begin your output directly without any introductory sentence or summary phrase. "
            
        text_prompt = text
        
        try:
           
            image_base64 = None
            
            # Call Gemini API
            response = call_gemini_api(
                text_prompt,
                image_base64,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (error_msg,)
            
#################################################

class GeminiVision:
    
    CATEGORY = main_cetegory() + "/LLM"
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail."
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
                    "default": "gemini-2.5-flash"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "system_instruction": ("ARTHAINSTRUCT", {
                    "forceInput": True,
                    "multiline": True
                }),                
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, image, text_prompt, api_key, model, max_tokens, temperature=0.7, system_instruction=""):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        if not system_instruction:
            
            system_instruction = load_agent("vision")
                
        try:
            # Convert tensor directly to base64
            pil_image = tensor_to_pil_image(image)
            
            # Call Gemini API
            response = call_gemini_api(
                text_prompt,
                pil_image,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
            
            response = response.replace("*", "")
            response = response.replace("#", "")
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (error_msg,)
            
#################################################

class GeminiMotion:
    
    CATEGORY = main_cetegory() + "/LLM"
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this video in detail."
                }),
                "resize": (["None", "480p", "360p", "240p"], {
                    "default": "None"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
                    "default": "gemini-2.5-flash"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "system_instruction": ("ARTHAINSTRUCT", {
                    "forceInput": True,
                    "multiline": True
                }),                
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, image, text_prompt, resize, api_key, model, max_tokens, temperature=0.7, system_instruction=""):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        if not system_instruction:
            
            system_instruction = load_agent("motion")
            
        try:
            
            pil_images = []
            
            for img in image:
                
                imp = tensor_to_pil_image(img)
                
                if resize == "480p":
                    
                    imp = resize_image_shortest(imp, 480)
                
                if resize == "360p":
                    
                    imp = resize_image_shortest(imp, 360)
                    
                if resize == "240p":
                    
                    imp = resize_image_shortest(imp, 240)
                
                pil_images.append(imp)
                
            # Call Gemini API
            response = call_gemini_api(
                text_prompt,
                pil_images,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
            
            response = response.replace("*", "")
            response = response.replace("#", "")
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (error_msg,)

#################################################

class GeminiPrompter:
    
    CATEGORY = main_cetegory() + "/LLM"
    
    DESCRIPTION = "Gemini Prompter Nodes's objective is to enrich the content of your prompt."
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cat with a hat.",
                    "tooltip": ""
                }),
                "media": (["IMAGE", "VIDEO"], {
                    "default": "IMAGE"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
                    "default": "gemini-2.5-flash"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "system_instruction": ("ARTHAINSTRUCT", {
                    "forceInput": True,
                    "multiline": True
                }),                
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, text_prompt, media, api_key, model, max_tokens, temperature=0.7, system_instruction=""):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        if not system_instruction:
            
            if media == "IMAGE":
            
                system_instruction = load_agent("enrich_image")
                
            if media == "VIDEO":
            
                system_instruction = load_agent("enrich_video")
        
        try:
           
            image_base64 = None
            
            # Call Gemini API
            response = call_gemini_api(
                text_prompt,
                image_base64,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (error_msg,)
            
#################################################

class GeminiCondense:
    
    CATEGORY = main_cetegory() + "/LLM"
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": ""
                }),
                "max_words": ("INT", {
                    "default": 400,
                    "min": 1,
                    "max": 10000,
                    "step": 10,
                    "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
                    "default": "gemini-2.5-flash"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words."
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),              
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, text_prompt, max_words, api_key, model, max_tokens, temperature=0.7):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
                
        system_instruction = "Role: You are a master of conciseness, understanding that every word carries weight and "
        system_instruction += "that unnecessary complexity can confuse AI image generators. Your expertise includes a "
        system_instruction += "deep knowledge of artistic styles, lighting techniques, and compositional theories, "
        system_instruction += "allowing you to select the most impactful descriptive language. "
        system_instruction += "You do not just shorten text; you strategically re-engineer it "
        system_instruction += "for maximum clarity and artistic output. "
        system_instruction += "Begin your output directly without any introductory sentence or summary phrase. \n\n"
        
        system_instruction += "Task: Your task is to take a user-submitted image generation prompt of any length "
        system_instruction += "and condense it into a clear, structured, and highly effective prompt of no "
        system_instruction += "more than " + str(max_words) + " words. "
        system_instruction += "The final output must be optimized for current text-to-image AI models. \n\n"
        
        try:
           
            image_base64 = None
            
            # Call Gemini API
            response = call_gemini_api(
                text_prompt,
                image_base64,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (error_msg,)

#################################################            

# Required mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Gemini Question": GeminiQuestion,
    "Gemini Operation": GeminiOperation,
    "Gemini Translate": GeminiTranslate,
    "Gemini Prompter": GeminiPrompter,
    "Gemini Condense": GeminiCondense,
    "Gemini Vision": GeminiVision,
    "Gemini Motion": GeminiMotion     
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini Question": node_prefix() + " Gemini Question",
    "Gemini Operation": node_prefix() + " Gemini Operation",
    "Gemini Translate": node_prefix() + " Gemini Translate",
    "Gemini Prompter": node_prefix() + " Gemini Prompter", 
    "Gemini Condense": node_prefix() + " Gemini Condense", 
    "Gemini Vision": node_prefix() + " Gemini Vision",
    "Gemini Motion": node_prefix() + " Gemini Motion"      
}