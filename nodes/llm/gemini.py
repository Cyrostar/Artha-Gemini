import os
import json
import random

from PIL import Image
import numpy as np

import folder_paths

from ...core.node import node_path, node_prefix, main_cetegory
from ...core.api import load_api_key
from ...core.llm import call_gemini_text_api
from ...core.llm import call_gemini_image_api
from ...core.llm import call_gemini_tts_api
from ...core.llm import gemini_api_parameters
from ...core.llm import load_agent
from ...core.img import tensor_to_pil_image, resize_image_shortest

################################################# 

class GeminiQuestion:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Gemini Question node can be used to ask questions "
    DESCRIPTION += "though the context will not be preserved as "
    DESCRIPTION += "this node is not suitable for dialogue purposes."
    
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
                **gemini_api_parameters(),            
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, question, api_key, model, max_tokens, temperature):
            
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
            
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
################################################# 

class GeminiOperation:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Gemini Operation node can be used to make "
    DESCRIPTION += "changes on the source text like replacing, "
    DESCRIPTION += "removing parts and other text operations."
    
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
                **gemini_api_parameters(),             
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, source, instruction, api_key, model, max_tokens, temperature):
            
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
            
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
################################################# 

class GeminiTranslate:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Gemini Translate node can be used to translate "
    DESCRIPTION += "one language to another."
    
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
                **gemini_api_parameters(),             
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, text, lang_from, lang_to, api_key, model, max_tokens, temperature):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
                  
        system_instruction = "Translate the given text from " + lang_from + " to " + lang_to + "."
        system_instruction += "Begin your output directly without any introductory sentence or summary phrase. "
            
        text_prompt = text
        
        try:
            
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
#################################################

class GeminiImagen:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Gemini Imagen node is for image generation and modification."
    
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
                **gemini_api_parameters(model="image"),  
                "modify_image": ("BOOLEAN", {"default": False}),
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "system_instruction": ("ARTHAINSTRUCT", {
                    "forceInput": True,
                    "multiline": True
                }),                
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
            tensor, response = call_gemini_image_api(
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

class GeminiSpeech:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Generates speech from text using the Gemini TTS model."
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "speech.json")
    
    with open(json_path, 'r') as file:
        
        speech = json.load(file)
        
    speakers = speech['Speakers']
    language = speech['Languages']
    
    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cat with a hat"
                }),
                "voice": (list(self.speakers.keys()), { "default": "Kore" }),
                **gemini_api_parameters(model="tts"),  
            },
        }
           
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "artha_main"

    def artha_main(self, text_prompt, voice, api_key, model, max_tokens, temperature):
        
        audio = None
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (audio,)       
                                       
        # Call Gemini API
        audio = call_gemini_tts_api(
            text_prompt,
            voice,
            api_key, 
            model,
            temperature
        )
        
        return (audio,)
                      
#################################################

class GeminiVision:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Gemini Vision node outputs a rich description "
    DESCRIPTION += "of the input image."
    
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
                **gemini_api_parameters(),
            },
            "optional": {
                "system_instruction": ("ARTHAINSTRUCT", {
                    "forceInput": True,
                    "multiline": True
                }),                
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, image, text_prompt, api_key, model, max_tokens, temperature, system_instruction=""):
            
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
            kwargs = {
                'text_prompt'       : text_prompt,
                'image'             : pil_image,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            response = response.translate(str.maketrans("", "", "*#"))
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
#################################################

class GeminiMotion:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Gemini Vision node outputs a rich description "
    DESCRIPTION += "of the input video. If the video is too large "
    DESCRIPTION += "in file size, the built-in resizer can be used "
    DESCRIPTION += "to lower the size to a reasonable level."
    
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
                **gemini_api_parameters(),
            },
            "optional": {
                "system_instruction": ("ARTHAINSTRUCT", {
                    "forceInput": True,
                    "multiline": True
                }),                
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, image, text_prompt, resize, api_key, model, max_tokens, temperature, system_instruction=""):
            
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
                
                # Convert tensor directly to base64
                imp = tensor_to_pil_image(img)
                
                if resize == "480p":
                    
                    imp = resize_image_shortest(imp, 480)
                
                if resize == "360p":
                    
                    imp = resize_image_shortest(imp, 360)
                    
                if resize == "240p":
                    
                    imp = resize_image_shortest(imp, 240)
                
                pil_images.append(imp)                
           
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'image'             : pil_images,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            response = response.translate(str.maketrans("", "", "*#"))
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)

#################################################

class GeminiPrompter:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    
    DESCRIPTION = "Gemini Prompter node enriches the content of your prompt."
    
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
                **gemini_api_parameters(),
            },
            "optional": {
                "system_instruction": ("ARTHAINSTRUCT", {
                    "forceInput": True,
                    "multiline": True
                }),                
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, text_prompt, media, api_key, model, max_tokens, temperature, system_instruction=""):
            
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
            
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
#################################################

class GeminiCondense:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Gemini Question node condenses a propt to a target word count preserving the concept."
    
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
                **gemini_api_parameters(),             
            },                
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    
    def artha_main(self, text_prompt, max_words, api_key, model, max_tokens, temperature):
            
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
            
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
#################################################
####################PROFILE######################
#################################################

class GeminiPortrait:
     
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes a character. "
    DESCRIPTION += "When no image connected, output is crafted by the parameters. "
    DESCRIPTION += "If image input is active, prompt is crafted from the image details. "
    DESCRIPTION += "If image input is active and reconstruct is true, prompt is crafted from " 
    DESCRIPTION += "the image details but parameters will override the attributes fetched from the image."
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "profile.json")
    
    with open(json_path, 'r') as file:
        
        character = json.load(file)   
    
    def __init__(self):       
        pass
    
    @classmethod
    def INPUT_TYPES(self):
        
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Construct a prompt describing a character in detail."
                }),
                "identity": (["FEMININE", "MASCULINE"], { "default": "FEMININE" }),
                "framing": (["headshot", "portrait", "medium shot", "wide shot", "full body shot"], {
                    "default": "portrait"
                }),
                **gemini_api_parameters(),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "use_image": ("BOOLEAN", {"default": False}),
                "reconstruct": ("BOOLEAN", {"default": False}),
                "image": (sorted(files), {"image_upload": True})
            },
            "optional": {
                "face": ("ARTHAFACE", {"forceInput":True}),
                "body": ("ARTHABODY", {"forceInput":True}),
                "form": ("ARTHAFORM", {"forceInput":True}), 
                "cloth": ("ARTHACLOTH", {"forceInput":True}), 
                "makeup": ("ARTHAMAKEUP", {"forceInput":True}),                                                                        
            },          
        }
        
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "traits",)
    FUNCTION = "artha_main"
    
    def artha_main(self, text_prompt, identity, framing, seed, use_image, reconstruct, image, api_key, model, max_tokens, temperature, **kwargs):
        
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
                       
                        
        face    = kwargs.get('face',    None)
        body    = kwargs.get('body',    None)
        form    = kwargs.get('form',    None)
        makeup  = kwargs.get('makeup',  None)
        cloth   = kwargs.get('cloth',   None)
               
               
        system_instruction = "You are a professional expert in facial aesthetics, makeup, hairstyling, structural anatomy and fashion. \n\n" 
        
        if(identity == "FEMININE"):
            
            system_instruction += "Your task is to generate a clear and visually rich description of a female, "
            
        else:
            
            system_instruction += "Your task is to generate a clear and visually rich description of a male, "
        
        system_instruction += "based on a given set of attributes. \n\n"
        
        system_instruction += "Combine the provided attributes into a cohesive, elegant, and realistic description. "
        system_instruction += "Keep the tone simple, precise, and meaningful. "
        
        if(identity == "FEMININE"):
            
            system_instruction += "Use the pronoun She or Her for the subject. "
            
        else:
            
            system_instruction += "Use the pronoun He or His for the subject. "
            
        system_instruction += "Ensure the description is specific, vivid, and unambiguous. " 
        system_instruction += "Avoid redundant phrases and excessive adjectives. "
        system_instruction += "Optimize the prompt for a " + framing + " framed photo. "
        system_instruction += "Begin your output directly without any introductory sentence or summary phrase. \n\n"
                
        property_list = ""
        
        if face:
            
            if isinstance(face, dict):
            
                for key, value in face.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    if identity == "FEMININE" and key == 'HAIR_STYLE_MAS':
                        
                        continue
                        
                    if identity == "MASCULINE" and key == 'HAIR_STYLE_FEM':
                        
                        continue
                        
                    if key == "HAIR_STYLE_FEM": key = "HAIR_STYLE"
                    if key == "HAIR_STYLE_MAS": key = "HAIR_STYLE"                  
                        
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n" 
                    
            else:
                
                property_list += face
                
        if body:
            
            if isinstance(body, dict):
            
                for key, value in body.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += body                   
                
        if form:
            
            if isinstance(form, dict):
            
                for key, value in form.items():
                    
                    if value == 'NONE':
                        
                        continue
                        
                    property_list += "- FORM " + key.replace('_', ' ') + ": " + value.upper() + "\n" 
                        
            else:
                
                property_list += form      
    
        if cloth:
            
            property_list += "\n"
            property_list += cloth
            property_list += "\n"
        
        if makeup:
            
            property_list += "\n"
            property_list += makeup
            property_list += "\n"
 
 
        if image and use_image and not reconstruct :
                        
            system_instruction += "Use the provided image as your main reference. "
            system_instruction += "Carefully analyze it and describe the features listed below in detail.\n\n"
            
            system_instruction += "Features to extract: \n"
            system_instruction += "- Facial features \n"
            system_instruction += "- Body characteristics \n"
            system_instruction += "- Fitness indicators \n"
            system_instruction += "- Clothing style and appearance \n"
            system_instruction += "- Makeup details \n"
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature,
                        'seed'              : seed
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, property_list)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, property_list)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, property_list)
                           
        elif image and use_image and reconstruct :
            
            system_instruction += "Use the provided image as your main reference. "
            system_instruction += "Carefully analyze it and describe the features listed below in detail.\n\n"
            
            system_instruction += "Features to extract: \n"
            system_instruction += "- Facial features \n"
            system_instruction += "- Body characteristics \n"
            system_instruction += "- Fitness indicators \n"
            system_instruction += "- Clothing style and appearance \n"
            system_instruction += "- Makeup details \n\n"
            
            system_instruction += "Change the attributes you identified from the image with the ones that are listed " 
            system_instruction += "in the property list if there is a confliction. For example if the eye color you "
            system_instruction += "identified from the image is green but property list says it is blue, use the blue color. "
            system_instruction += "Also add the properties from the list which is absent from the input image. \n\n"
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature,
                        'seed'              : seed
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, property_list)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, property_list)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, property_list)
            
        else:
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            system_instruction += "\n"
                       
            system_instruction += "If property list is empty return an empty response. \n\n"
               
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature,
                    'seed'              : seed
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, property_list)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, property_list)
                
#################################################

class GeminiFace:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes the face structure of the subject."
             
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "profile.json")
    
    with open(json_path, 'r') as file:
        
        character = json.load(file)
        
    HEAD_TYPES              = character['HEAD']['HEAD_TYPES']
    HAIR_COLORS             = character['HAIR']['HAIR_COLORS'] 
    HAIR_LENGTHS            = character['HAIR']['HAIR_LENGTHS'] 
    HAIR_STYLES_FEM         = character['HAIR']['HAIR_STYLES_FEMININE']
    HAIR_STYLES_MAS         = character['HAIR']['HAIR_STYLES_MASCULINE']
    FACE_APPEAL             = character['FACE']['FACE_APPEAL']  
    FACE_AGE                = character['FACE']['FACE_AGE']     
    FACE_SHAPES             = character['FACE']['FACE_SHAPES']
    FACE_EYEBROW_TYPES      = character['FACE']['EYEBROW_TYPES']
    FACE_EYEBROW_SHAPES     = character['FACE']['EYEBROW_SHAPES']
    FACE_EYE_TYPES          = character['FACE']['EYE_TYPES']     
    FACE_EYE_SIZES          = character['FACE']['EYE_SIZES'] 
    FACE_EYE_COLORS         = character['FACE']['EYE_COLORS'] 
    FACE_NOSE_TYPES         = character['FACE']['NOSE_TYPES'] 
    FACE_LIP_TYPES          = character['FACE']['LIP_TYPES'] 
    FACE_LIP_COLORS         = character['FACE']['LIP_COLORS']
    FACE_EAR_TYPES          = character['FACE']['EAR_TYPES']
    FACE_CHEEK_TYPES        = character['FACE']['CHEEK_TYPES']   
    FACE_CHIN_TYPES         = character['FACE']['CHIN_TYPES']
    
    @classmethod
    def INPUT_TYPES(self):
        
        comfyui_input_dir = folder_paths.get_input_directory()
        image_file_list = [f for f in os.listdir(comfyui_input_dir) if os.path.isfile(os.path.join(comfyui_input_dir, f))]
    
        return {
            "required": {
                "head_type":            (["NONE"] + list(self.HEAD_TYPES.keys()),           { "default": "NONE" }),
                "hair_color":           (["NONE"] + list(self.HAIR_COLORS.keys()),          { "default": "NONE" }),
                "hair_length":          (["NONE"] + list(self.HAIR_LENGTHS.keys()),         { "default": "NONE" }),
                "hair_style_fem":       (["NONE"] + list(self.HAIR_STYLES_FEM.keys()),      { "default": "NONE" }),
                "hair_style_mas":       (["NONE"] + list(self.HAIR_STYLES_MAS.keys()),      { "default": "NONE" }),
                "face_appeal":          (["NONE"] + list(self.FACE_APPEAL.keys()),          { "default": "NONE" }),
                "face_age":             (["NONE"] + list(self.FACE_AGE.keys()),             { "default": "NONE" }),
                "face_shape":           (["NONE"] + list(self.FACE_SHAPES.keys()),          { "default": "NONE" }),
                "face_eyebrow_type":    (["NONE"] + list(self.FACE_EYEBROW_TYPES.keys()),   { "default": "NONE" }),
                "face_eyebrow_shape":   (["NONE"] + list(self.FACE_EYEBROW_SHAPES.keys()),  { "default": "NONE" }),
                "face_eye_type":        (["NONE"] + list(self.FACE_EYE_TYPES.keys()),       { "default": "NONE" }),
                "face_eye_size":        (["NONE"] + list(self.FACE_EYE_SIZES.keys()),       { "default": "NONE" }),
                "face_eye_color":       (["NONE"] + list(self.FACE_EYE_COLORS.keys()),      { "default": "NONE" }),
                "face_nose_type":       (["NONE"] + list(self.FACE_NOSE_TYPES.keys()),      { "default": "NONE" }),
                "face_lip_type":        (["NONE"] + list(self.FACE_LIP_TYPES.keys()),       { "default": "NONE" }),
                "face_lip_color":       (["NONE"] + list(self.FACE_LIP_COLORS.keys()),      { "default": "NONE" }),
                "face_ear_type":        (["NONE"] + list(self.FACE_EAR_TYPES.keys()),       { "default": "NONE" }),
                "face_cheek_type":      (["NONE"] + list(self.FACE_CHEEK_TYPES.keys()),     { "default": "NONE" }),
                "face_chin_type":       (["NONE"] + list(self.FACE_CHIN_TYPES.keys()),      { "default": "NONE" }),
                **gemini_api_parameters(),
                "randomize":            ("BOOLEAN", {"default": False}),
                "use_image":            ("BOOLEAN", {"default": False}),
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHAFACE",)
    RETURN_NAMES = ("face",)
    FUNCTION = "artha_main"

    def artha_main(self, head_type, hair_color, hair_length, hair_style_fem, hair_style_mas, face_appeal, face_age, face_shape, face_eyebrow_type, face_eyebrow_shape, face_eye_type, face_eye_size, face_eye_color, face_nose_type, face_lip_type, face_lip_color, face_ear_type, face_cheek_type, face_chin_type, randomize, use_image, image, api_key, model, max_tokens, temperature, **kwargs):
              
        if image and use_image :
            
            response = ""
                     
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                return (response,)
            
            text_prompt = "Describe the face in detail."
            
            system_instruction = load_agent("face")
            
            image_path = folder_paths.get_annotated_filepath(image)
                            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response,)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response,)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                       
        else:
            
            face_dict = {}
                         
            if randomize:
            
                face_dict['HEAD_TYPE']           = random.choice([key for key in self.HEAD_TYPES.keys()          if key != 'NONE'])
                face_dict['HAIR_COLOR']          = random.choice([key for key in self.HAIR_COLORS.keys()         if key != 'NONE'])
                face_dict['HAIR_LENGTH']         = random.choice([key for key in self.HAIR_LENGTHS.keys()        if key != 'NONE'])
                face_dict['HAIR_STYLE_FEM']      = random.choice([key for key in self.HAIR_STYLES_FEM.keys()     if key != 'NONE'])
                face_dict['HAIR_STYLE_MAS']      = random.choice([key for key in self.HAIR_STYLES_MAS.keys()     if key != 'NONE'])
                face_dict['FACE_APPEAL']         = random.choice([key for key in self.FACE_APPEAL.keys()         if key != 'NONE'])
                face_dict['FACE_AGE']            = random.choice([key for key in self.FACE_AGE.keys()            if key != 'NONE'])
                face_dict['FACE_SHAPE']          = random.choice([key for key in self.FACE_SHAPES.keys()         if key != 'NONE'])
                face_dict['FACE_EYEBROW_TYPE']   = random.choice([key for key in self.FACE_EYEBROW_TYPES.keys()  if key != 'NONE'])
                face_dict['FACE_EYEBROW_SHAPE']  = random.choice([key for key in self.FACE_EYEBROW_SHAPES.keys() if key != 'NONE'])
                face_dict['FACE_EYE_TYPE']       = random.choice([key for key in self.FACE_EYE_TYPES.keys()      if key != 'NONE'])
                face_dict['FACE_EYE_SIZE']       = random.choice([key for key in self.FACE_EYE_SIZES.keys()      if key != 'NONE'])
                face_dict['FACE_EYE_COLOR']      = random.choice([key for key in self.FACE_EYE_COLORS.keys()     if key != 'NONE'])
                face_dict['FACE_NOSE_TYPE']      = random.choice([key for key in self.FACE_NOSE_TYPES.keys()     if key != 'NONE'])
                face_dict['FACE_LIP_TYPE']       = random.choice([key for key in self.FACE_LIP_TYPES.keys()      if key != 'NONE'])
                face_dict['FACE_LIP_COLOR']      = random.choice([key for key in self.FACE_LIP_COLORS.keys()     if key != 'NONE'])
                face_dict['FACE_EAR_TYPE']       = random.choice([key for key in self.FACE_EAR_TYPES.keys()      if key != 'NONE'])
                face_dict['FACE_CHEEK_TYPE']     = random.choice([key for key in self.FACE_CHEEK_TYPES.keys()    if key != 'NONE'])
                face_dict['FACE_CHIN_TYPE']      = random.choice([key for key in self.FACE_CHIN_TYPES.keys()     if key != 'NONE'])
                
            else:
                
                face_dict['HEAD_TYPE']           = head_type
                face_dict['HAIR_COLOR']          = hair_color
                face_dict['HAIR_LENGTH']         = hair_length
                face_dict['HAIR_STYLE_FEM']      = hair_style_fem
                face_dict['HAIR_STYLE_MAS']      = hair_style_mas
                face_dict['FACE_APPEAL']         = face_appeal
                face_dict['FACE_AGE']            = face_age
                face_dict['FACE_SHAPE']          = face_shape
                face_dict['FACE_EYEBROW_TYPE']   = face_eyebrow_type
                face_dict['FACE_EYEBROW_SHAPE']  = face_eyebrow_shape
                face_dict['FACE_EYE_TYPE']       = face_eye_type
                face_dict['FACE_EYE_SIZE']       = face_eye_size
                face_dict['FACE_EYE_COLOR']      = face_eye_color
                face_dict['FACE_NOSE_TYPE']      = face_nose_type
                face_dict['FACE_LIP_TYPE']       = face_lip_type
                face_dict['FACE_LIP_COLOR']      = face_lip_color
                face_dict['FACE_EAR_TYPE']       = face_ear_type
                face_dict['FACE_CHEEK_TYPE']     = face_cheek_type
                face_dict['FACE_CHIN_TYPE']      = face_chin_type 
            
            return (face_dict,) 
  
#################################################

class GeminiBody:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes the body structure of the subject."
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "profile.json")
    
    with open(json_path, 'r') as file:
        
        character = json.load(file)
        
    BODY_TYPES              = character['BODY']['BODY_TYPES']
    BODY_HEIGHT             = character['BODY']['HEIGHT']
    BODY_WEIGHT             = character['BODY']['WEIGHT'] 
    BODY_BUILD              = character['BODY']['BUILD']
    BODY_FRAME              = character['BODY']['FRAME']     
    BODY_SHOULDER           = character['BODY']['SHOULDER']
    BODY_CHEST              = character['BODY']['CHEST'] 
    BODY_BREASTS            = character['BODY']['BREASTS']
    BODY_TORSO              = character['BODY']['TORSO']
    BODY_WAIST              = character['BODY']['WAIST']
    BODY_HIP                = character['BODY']['HIP'] 
    BODY_LEGS               = character['BODY']['LEGS']
    BODY_SKIN_TONE          = character['BODY']['SKIN_TONE']
    BODY_POSTURE            = character['BODY']['POSTURE']
    
    @classmethod
    def INPUT_TYPES(self):
        
        comfyui_input_dir = folder_paths.get_input_directory()
        image_file_list = [f for f in os.listdir(comfyui_input_dir) if os.path.isfile(os.path.join(comfyui_input_dir, f))]
        
        return {
            "required": {
                "body_type":            (["NONE"] + list(self.BODY_TYPES.keys()),           { "default": "NONE" }),
                "body_height":          (["NONE"] + list(self.BODY_HEIGHT.keys()),          { "default": "NONE" }),
                "body_weight":          (["NONE"] + list(self.BODY_WEIGHT.keys()),          { "default": "NONE" }),
                "body_build":           (["NONE"] + list(self.BODY_BUILD.keys()),           { "default": "NONE" }),
                "body_frame":           (["NONE"] + list(self.BODY_FRAME.keys()),           { "default": "NONE" }),
                "body_shoulder":        (["NONE"] + list(self.BODY_SHOULDER.keys()),        { "default": "NONE" }),
                "body_chest":           (["NONE"] + list(self.BODY_CHEST.keys()),           { "default": "NONE" }),
                "body_breasts":         (["NONE"] + list(self.BODY_BREASTS.keys()),         { "default": "NONE" }),
                "body_torso":           (["NONE"] + list(self.BODY_TORSO.keys()),           { "default": "NONE" }),
                "body_waist":           (["NONE"] + list(self.BODY_WAIST.keys()),           { "default": "NONE" }),
                "body_hip":             (["NONE"] + list(self.BODY_HIP.keys()),             { "default": "NONE" }),
                "body_legs":            (["NONE"] + list(self.BODY_LEGS.keys()),            { "default": "NONE" }),
                "body_skin_tone":       (["NONE"] + list(self.BODY_SKIN_TONE.keys()),       { "default": "NONE" }),
                "body_posture":         (["NONE"] + list(self.BODY_POSTURE.keys()),         { "default": "NONE" }),
                **gemini_api_parameters(),
                "randomize":            ("BOOLEAN", {"default": False}),
                "use_image":            ("BOOLEAN", {"default": False}),
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHABODY",)
    RETURN_NAMES = ("body",)
    FUNCTION = "artha_main"

    def artha_main(self, body_type, body_height, body_weight, body_build, body_frame, body_shoulder, body_chest, body_breasts, body_torso, body_waist, body_hip, body_legs, body_skin_tone, body_posture, randomize, use_image, image, api_key, model, max_tokens, temperature, **kwargs):
    
        if image and use_image :
            
            response = ""
                     
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response,)
            
            text_prompt = "Describe the body in detail."
            
            system_instruction = load_agent("body")
            
            image_path = folder_paths.get_annotated_filepath(image)
                            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response,)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response,)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                       
        else:
            
            body_dict = {}
            
            if randomize:
                
                body_dict['BODY_TYPES']     = random.choice([key for key in self.BODY_TYPES.keys()      if key != 'NONE'])
                body_dict['BODY_HEIGHT']    = random.choice([key for key in self.BODY_HEIGHT.keys()     if key != 'NONE'])
                body_dict['BODY_WEIGHT']    = random.choice([key for key in self.BODY_WEIGHT.keys()     if key != 'NONE'])
                body_dict['BODY_BUILD']     = random.choice([key for key in self.BODY_BUILD.keys()      if key != 'NONE'])
                body_dict['BODY_FRAME']     = random.choice([key for key in self.BODY_FRAME.keys()      if key != 'NONE'])
                body_dict['BODY_SHOULDER']  = random.choice([key for key in self.BODY_SHOULDER.keys()   if key != 'NONE'])
                body_dict['BODY_CHEST']     = random.choice([key for key in self.BODY_CHEST.keys()      if key != 'NONE'])
                body_dict['BODY_BREASTS']   = random.choice([key for key in self.BODY_BREASTS.keys()    if key != 'NONE'])
                body_dict['BODY_TORSO']     = random.choice([key for key in self.BODY_TORSO.keys()      if key != 'NONE'])
                body_dict['BODY_WAIST']     = random.choice([key for key in self.BODY_WAIST.keys()      if key != 'NONE'])
                body_dict['BODY_HIP']       = random.choice([key for key in self.BODY_HIP.keys()        if key != 'NONE'])
                body_dict['BODY_LEGS']      = random.choice([key for key in self.BODY_LEGS.keys()       if key != 'NONE'])
                body_dict['BODY_SKIN_TONE'] = random.choice([key for key in self.BODY_SKIN_TONE.keys()  if key != 'NONE'])
                body_dict['BODY_POSTURE']   = random.choice([key for key in self.BODY_POSTURE.keys()    if key != 'NONE'])
                        
            else:
    
                body_dict['BODY_TYPES']     = body_type     
                body_dict['BODY_HEIGHT']    = body_height   
                body_dict['BODY_WEIGHT']    = body_weight   
                body_dict['BODY_BUILD']     = body_build    
                body_dict['BODY_FRAME']     = body_frame    
                body_dict['BODY_SHOULDER']  = body_shoulder 
                body_dict['BODY_CHEST']     = body_chest    
                body_dict['BODY_BREASTS']   = body_breasts  
                body_dict['BODY_TORSO']     = body_torso    
                body_dict['BODY_WAIST']     = body_waist    
                body_dict['BODY_HIP']       = body_hip      
                body_dict['BODY_LEGS']      = body_legs     
                body_dict['BODY_SKIN_TONE'] = body_skin_tone
                body_dict['BODY_POSTURE']   = body_posture  
                    
            return (body_dict,) 
  
#################################################

class GeminiForm:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes fitness of the subject."
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "profile.json")
    
    with open(json_path, 'r') as file:
        
        character = json.load(file)
        
    FORM = character['FORM']
    
    @classmethod
    def INPUT_TYPES(self):
        
        comfyui_input_dir = folder_paths.get_input_directory()
        image_file_list = [f for f in os.listdir(comfyui_input_dir) if os.path.isfile(os.path.join(comfyui_input_dir, f))]
        
        return {
            "required": {
                "chest":        ("BOOLEAN", {"default": False}),
                "shoulders":    ("BOOLEAN", {"default": False}),
                "arms":         ("BOOLEAN", {"default": False}),
                "biceps":       ("BOOLEAN", {"default": False}),
                "triceps":      ("BOOLEAN", {"default": False}),
                "forearms":     ("BOOLEAN", {"default": False}),
                "abs":          ("BOOLEAN", {"default": False}),
                "core":         ("BOOLEAN", {"default": False}),
                "obliques":     ("BOOLEAN", {"default": False}),
                "back":         ("BOOLEAN", {"default": False}),
                "lats":         ("BOOLEAN", {"default": False}),
                "traps":        ("BOOLEAN", {"default": False}),
                "legs":         ("BOOLEAN", {"default": False}),
                "quadriceps":   ("BOOLEAN", {"default": False}),
                "hamstrings":   ("BOOLEAN", {"default": False}),
                "calves":       ("BOOLEAN", {"default": False}),
                "glutes":       ("BOOLEAN", {"default": False}),
                **gemini_api_parameters(),
                "use_image": ("BOOLEAN", {"default": False}),
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHAFORM",)
    RETURN_NAMES = ("form",)
    FUNCTION = "artha_main"

    def artha_main(self, chest, shoulders, arms, biceps, triceps, forearms, abs, core, obliques, back, lats, traps, legs, quadriceps, hamstrings, calves, glutes, use_image, image, api_key, model, max_tokens, temperature, **kwargs):
        
        if image and use_image :
            
            response = ""
                     
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response,)
            
            text_prompt = "Describe the fitness in detail."
            
            system_instruction = load_agent("form")
            
            image_path = folder_paths.get_annotated_filepath(image)
                            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response,)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response,)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                       
        else:
            
            form_dict = {}
                
            if chest      : form_dict['CHEST']      = self.FORM['CHEST']     
            if shoulders  : form_dict['SHOULDERS']  = self.FORM['SHOULDERS'] 
            if arms       : form_dict['ARMS']       = self.FORM['ARMS']      
            if biceps     : form_dict['BICEPS']     = self.FORM['BICEPS']   
            if triceps    : form_dict['TRICEPS']    = self.FORM['TRICEPS']  
            if forearms   : form_dict['FOREARMS']   = self.FORM['FOREARMS'] 
            if abs        : form_dict['ABS']        = self.FORM['ABS']      
            if core       : form_dict['CORE']       = self.FORM['CORE']     
            if obliques   : form_dict['OBLIQUES']   = self.FORM['OBLIQUES'] 
            if back       : form_dict['BACK']       = self.FORM['BACK']      
            if lats       : form_dict['LATS']       = self.FORM['LATS']     
            if traps      : form_dict['TRAPS']      = self.FORM['TRAPS']    
            if legs       : form_dict['LEGS']       = self.FORM['LEGS']     
            if quadriceps : form_dict['QUADRICEPS'] = self.FORM['QUADRICEPS']
            if hamstrings : form_dict['HAMSTRINGS'] = self.FORM['HAMSTRINGS']
            if calves     : form_dict['CALVES']     = self.FORM['CALVES']    
            if glutes     : form_dict['GLUTES']     = self.FORM['GLUTES']  
            
            return (form_dict,)

#################################################  

class GeminiCloth:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes the clothing of the subject."
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                **gemini_api_parameters(),
                "image": (sorted(files), {"image_upload": True}),                
            },
        }
           
    RETURN_TYPES = ("ARTHACLOTH",)
    RETURN_NAMES = ("cloth",)
    FUNCTION = "artha_main"

    def artha_main(self, image, api_key, model, max_tokens, temperature, **kwargs):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Identify the clothes and list each one."
        
        system_instruction = load_agent("cloth")
            
        image_path = folder_paths.get_annotated_filepath(image)
                            
        try:
            
            pil_image = Image.open(image_path).convert("RGB")
            
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response,)
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
            
#################################################

class GeminiMakeup:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes the make-up of the subject."
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                **gemini_api_parameters(),
                "image": (sorted(files), {"image_upload": True}),
            },
        }
           
    RETURN_TYPES = ("ARTHAMAKEUP",)
    RETURN_NAMES = ("makeup",)
    FUNCTION = "artha_main"

    def artha_main(self, image, api_key, model, max_tokens, temperature, **kwargs):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Identify the make-up of the face and list each element."
        
        system_instruction = load_agent("makeup")
            
        image_path = folder_paths.get_annotated_filepath(image)
                            
        try:
            
            pil_image = Image.open(image_path).convert("RGB")
            
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response,)
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
            
#################################################

class GeminiBackdrop:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes the background of an image."
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                **gemini_api_parameters(),
                "image": (sorted(files), {"image_upload": True}),                
            },
        }
           
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("backdrop",)
    FUNCTION = "artha_main"

    def artha_main(self, image, api_key, model, max_tokens, temperature, **kwargs):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Describe the background."
        
        system_instruction = load_agent("backdrop")
            
        image_path = folder_paths.get_annotated_filepath(image)
                            
        try:
            
            pil_image = Image.open(image_path).convert("RGB")
            
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response,)
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
               
#################################################
####################COMPOSE######################
#################################################

class GeminiCompose:
     
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes a character. "
    DESCRIPTION += "When no image connected, output is crafted by the parameters. "
    DESCRIPTION += "If image input is active, prompt is crafted from the image details. "
    DESCRIPTION += "If image input is active and reconstruct is true, prompt is crafted from " 
    DESCRIPTION += "the image details but parameters will override the attributes fetched from the image."
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")
    
    with open(json_path, 'r') as file:
        
        character = json.load(file)   
    
    def __init__(self):
         
        pass
    
    @classmethod
    def INPUT_TYPES(self):
        
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Construct a prompt describing an image composition in detail."
                }),
                **gemini_api_parameters(),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,}),
                "use_image": ("BOOLEAN", {"default": False}),
                "reconstruct": ("BOOLEAN", {"default": False}),
                "image": (sorted(files), {"image_upload": True})
            },
            "optional": {
                "subject": ("ARTHASUBJECT", {"forceInput":True}),
                "scene": ("ARTHASCENERY", {"forceInput":True}),                 
                "camera": ("ARTHACAM", {"forceInput":True}),  
                "light": ("ARTHALIGHT", {"forceInput":True}),
                "style": ("ARTHASTYLE", {"forceInput":True}),                 
            },          
        }
        
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "traits",)
    FUNCTION = "artha_main"
    
    def artha_main(self, text_prompt, api_key, model, max_tokens, temperature, seed, use_image, reconstruct, image, **kwargs):
        
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
                       
        subject   = kwargs.get('subject',   None)
        scene     = kwargs.get('scene',     None)         
        camera    = kwargs.get('camera',    None)
        light     = kwargs.get('light',     None)
        style     = kwargs.get('style',     None)
             
        system_instruction = "You are a professional expert in visual composition. \n\n" 
                
        system_instruction += "Your task is to generate a clear and visually rich description of an "        
        system_instruction += "image composition based on a given set of attributes. \n\n"
        
        system_instruction += "Combine the provided attributes into a cohesive, elegant, and realistic description. "
        system_instruction += "Keep the tone simple, precise, and meaningful. "           
        system_instruction += "Ensure the description is specific, vivid, and unambiguous. " 
        system_instruction += "Avoid redundant phrases and excessive adjectives. "
        system_instruction += "Begin your output directly without any introductory sentence or summary phrase. \n\n"
                
        property_list = ""
        
        if subject:
            
            property_list += subject
                       
        if scene:
            
            if property_list: property_list += "\n\n"  
            
            if isinstance(scene, dict):
            
                for key, value in scene.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += scene 
                
        if camera:
            
            if property_list: property_list += "\n\n" 
            
            if isinstance(camera, dict):
            
                for key, value in camera.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += camera 

        if light:
            
            if property_list: property_list += "\n\n" 
            
            if isinstance(light, dict):
            
                for key, value in light.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += light  

        if style:
            
            if property_list: property_list += "\n\n" 
            
            if isinstance(style, dict):
            
                for key, value in style.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += style                  

        if image and use_image and not reconstruct :
                        
            system_instruction += "Use the provided image as your main reference. "
            system_instruction += "Carefully analyze it and describe the features listed below in detail.\n\n"
            
            system_instruction += "Features to extract: \n"
            system_instruction += "- Subject(s) \n"
            system_instruction += "- Scene and background \n"
            system_instruction += "- Camera settings and framing \n"
            system_instruction += "- Light \n"
            system_instruction += "- Style \n"
            system_instruction += "- Atmosphere \n"
            system_instruction += "- Mood \n\n"
            
            system_instruction += "Include face, hair, makeup, body, fitness, pose, and clothing of the main subjects(s). " 
            system_instruction += "Also include the interactions if the subjects are more than one. \n\n" 
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature,
                        'seed'              : seed
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, property_list)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, property_list)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, property_list)
                
                
        elif image and use_image and reconstruct :
            
            system_instruction += "Use the provided image as your main reference. "
            system_instruction += "Carefully analyze it and describe the features listed below in detail.\n\n"
            
            system_instruction += "Features to extract: \n"
            system_instruction += "- Subject(s) \n"
            system_instruction += "- Scene and background \n"
            system_instruction += "- Camera settings and framing \n"
            system_instruction += "- Light \n"
            system_instruction += "- Style \n"
            system_instruction += "- Atmosphere \n"
            system_instruction += "- Mood \n\n"
            
            system_instruction += "Include face, hair, makeup, body, fitness, pose, and clothing of the main subjects(s). " 
            system_instruction += "Also include the interactions if the subjects are more than one. \n\n" 
            
            system_instruction += "Change the attributes you identified from the image with the ones that are listed " 
            system_instruction += "in the property list if there is a confliction. For example if the eye color you "
            system_instruction += "identified from the image is green but property list says it is blue, use the blue color. "
            system_instruction += "Also add the properties from the list which is absent from the input image. \n\n"
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature,
                        'seed'              : seed
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, property_list)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, property_list)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, property_list)
            
        else:
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            system_instruction += "\n"
                       
            system_instruction += "If property list is empty return an empty response. \n\n"
               
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature,
                    'seed'              : seed
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, property_list)               
            
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, property_list)
                
#################################################

class GeminiSubject:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Subject settings for image or video prompting."
    DESCRIPTION += "If use image is selected the setting are fetched "
    DESCRIPTION += "from the uploaded image by Gemini Vision. "
   
    extra_params = {
        "use_image": ("BOOLEAN", {"default": False}),
        "only_main": ("BOOLEAN", {"default": True}),
    }
        
    @classmethod
    def INPUT_TYPES(self):
        
        comfyui_input_dir = folder_paths.get_input_directory()
        image_file_list = [f for f in os.listdir(comfyui_input_dir) if os.path.isfile(os.path.join(comfyui_input_dir, f))]
        
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cat with a hat"
                }),
                **gemini_api_parameters(),
                **self.extra_params,
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHASUBJECT",)
    RETURN_NAMES = ("subject",)
    FUNCTION = "artha_main"

    def artha_main(self, text_prompt, api_key, model, max_tokens, temperature, use_image, only_main, image, **kwargs):
    
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        if image and use_image :
            
            text_prompt = "Describe the subject in detail."
            
            if only_main:
            
                system_instruction = load_agent("subject")
                
            else:
            
                system_instruction = load_agent("subjects")
                           
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response,)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response,)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                       
        else:
            
            response = text_prompt
            
            return (response,)
            
################################################# 

class GeminiScenery:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Scene settings for image or video prompting."
    DESCRIPTION += "If use image is selected the setting are fetched "
    DESCRIPTION += "from the uploaded image by Gemini Vision. "
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")
    
    with open(json_path, 'r') as file:
        
        scenery = json.load(file)
        
    markdown = "# SCENE OPTIONS"
    markdown += "\n\n"
    
    for k, v in scenery['SCENERY'].items():
        
        markdown += "## " + k
        markdown += "\n"
        
        if isinstance(v, dict):

            for kk, vv in v.items():
                markdown += "- " + kk + ": " + str(vv)
                markdown += "\n"
        else:
 
            markdown += "### " + str(v)
            markdown += "\n"
        
        markdown += "\n"        
        
    SCENERY_LANDSCAPES      = scenery['SCENERY']['LANDSCAPES']
    SCENERY_URBAN           = scenery['SCENERY']['URBAN']
    SCENERY_INTERIOR        = scenery['SCENERY']['INTERIOR']
    SCENERY_FANTASY         = scenery['SCENERY']['FANTASY']
    SCENERY_FUTURISTIC      = scenery['SCENERY']['FUTURISTIC']
    SCENERY_ABSTRACT        = scenery['SCENERY']['ABSTRACT']
    SCENERY_MISCELLANEOUS   = scenery['SCENERY']['MISCELLANEOUS']

   
    extra_params = {
        "use_image": ("BOOLEAN", {"default": False}),
    }
        
    @classmethod
    def INPUT_TYPES(self):
        
        comfyui_input_dir = folder_paths.get_input_directory()
        image_file_list = [f for f in os.listdir(comfyui_input_dir) if os.path.isfile(os.path.join(comfyui_input_dir, f))]
        
        return {
            "required": {
                "landscapes":       (["NONE"] + list(self.SCENERY_LANDSCAPES      .keys()),   { "default": "NONE" }),
                "urban":            (["NONE"] + list(self.SCENERY_URBAN           .keys()),   { "default": "NONE" }),
                "interior":         (["NONE"] + list(self.SCENERY_INTERIOR        .keys()),   { "default": "NONE" }),
                "fantasy":          (["NONE"] + list(self.SCENERY_FANTASY         .keys()),   { "default": "NONE" }),
                "futuristic":       (["NONE"] + list(self.SCENERY_FUTURISTIC      .keys()),   { "default": "NONE" }),
                "abstract":         (["NONE"] + list(self.SCENERY_ABSTRACT        .keys()),   { "default": "NONE" }),
                "miscellaneous":    (["NONE"] + list(self.SCENERY_MISCELLANEOUS   .keys()),   { "default": "NONE" }),
                **gemini_api_parameters(),
                **self.extra_params,
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHASCENERY", "STRING")
    RETURN_NAMES = ("scene", "markdown")
    FUNCTION = "artha_main"

    def artha_main(self, landscapes, urban, interior, fantasy, futuristic, abstract, miscellaneous, api_key, model, max_tokens, temperature, use_image, image, **kwargs):
    
        response = ""
        
        markdown = self.markdown
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response, markdown)
        
        if image and use_image :
            
            text_prompt = "Describe the scene in detail."
            
            system_instruction = load_agent("scenery")
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:           
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                        
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, markdown)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, markdown)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, markdown)
                       
        else:
            
            text_prompt = ""
            
            if landscapes       != "NONE": text_prompt += self.SCENERY_LANDSCAPES[landscapes]
            if urban            != "NONE": text_prompt += self.SCENERY_URBAN[urban]
            if interior         != "NONE": text_prompt += self.SCENERY_INTERIOR[interior]
            if fantasy          != "NONE": text_prompt += self.SCENERY_FANTASY[fantasy]
            if futuristic       != "NONE": text_prompt += self.SCENERY_FUTURISTIC[futuristic]
            if abstract         != "NONE": text_prompt += self.SCENERY_ABSTRACT[abstract]
            if miscellaneous    != "NONE": text_prompt += self.SCENERY_MISCELLANEOUS[miscellaneous]
            
            system_instruction = load_agent("scene")

            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                    
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, markdown)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, markdown)
            
#################################################

class GeminiCamera:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Camera settings for image or video prompting."
    DESCRIPTION += "If use image is selected the setting are fetched "
    DESCRIPTION += "from the uploaded image by Gemini Vision. "
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")
    
    with open(json_path, 'r') as file:
        
        camera = json.load(file)
        
    markdown = "# CAMERA OPTIONS"
    markdown += "\n\n"
    
    for k, v in camera['CAMERA'].items():
        
        markdown += "## " + k
        markdown += "\n"
        
        if isinstance(v, dict):

            for kk, vv in v.items():
                markdown += "- " + kk + ": " + str(vv)
                markdown += "\n"
        else:
 
            markdown += "### " + str(v)
            markdown += "\n"
        
        markdown += "\n"
  
    TYPE            = camera['CAMERA']['TYPE']
    CONTEXT         = camera['CAMERA']['CONTEXT']     
    SHOT            = camera['CAMERA']['SHOT']
    LENSES          = camera['CAMERA']['LENSES']
    ANGLES          = camera['CAMERA']['ANGLES']
    MOTION          = camera['CAMERA']['MOTION']
    FOCAL_LENGTH    = camera['CAMERA']['FOCAL_LENGTH']
    DEPTH_OF_FIELD  = camera['CAMERA']['DEPTH_OF_FIELD']
    ASPECT_RATIO    = camera['CAMERA']['ASPECT_RATIO']
    SENSOR_TYPE     = camera['CAMERA']['SENSOR_TYPE']
    CAMERA_STYLE    = camera['CAMERA']['CAMERA_STYLE']
    FRAMING         = camera['CAMERA']['FRAMING']
    CAMERA_MODELS   = camera['CAMERA']['CAMERA_MODELS']
    FILMSTOCK       = camera['CAMERA']['FILMSTOCK']
    COLOR           = camera['CAMERA']['COLOR']
    TEXTURE         = camera['CAMERA']['TEXTURE']
    
    extra_params = {
        "randomize": ("BOOLEAN", {"default": False}),
        "use_image": ("BOOLEAN", {"default": False}),
    }
        
    @classmethod
    def INPUT_TYPES(self):
        
        comfyui_input_dir = folder_paths.get_input_directory()
        image_file_list = [f for f in os.listdir(comfyui_input_dir) if os.path.isfile(os.path.join(comfyui_input_dir, f))]
        
        return {
            "required": {
                "type":             (["NONE"] + list(self.TYPE              .keys()),   { "default": "NONE" }),
                "context":          (["NONE"] + list(self.CONTEXT           .keys()),   { "default": "NONE" }),
                "shot":             (["NONE"] + list(self.SHOT              .keys()),   { "default": "NONE" }),
                "lenses":           (["NONE"] + list(self.LENSES            .keys()),   { "default": "NONE" }),
                "angles":           (["NONE"] + list(self.ANGLES            .keys()),   { "default": "NONE" }),
                "motion":           (["NONE"] + list(self.MOTION            .keys()),   { "default": "NONE" }),
                "focal_length":     (["NONE"] + list(self.FOCAL_LENGTH      .keys()),   { "default": "NONE" }),
                "depth_of_field":   (["NONE"] + list(self.DEPTH_OF_FIELD    .keys()),   { "default": "NONE" }),
                "aspect_ratio":     (["NONE"] + list(self.ASPECT_RATIO      .keys()),   { "default": "NONE" }),
                "sensor_type":      (["NONE"] + list(self.SENSOR_TYPE       .keys()),   { "default": "NONE" }),
                "camera_style":     (["NONE"] + list(self.CAMERA_STYLE      .keys()),   { "default": "NONE" }),
                "framing":          (["NONE"] + list(self.FRAMING           .keys()),   { "default": "NONE" }),
                "camera_models":    (["NONE"] + list(self.CAMERA_MODELS     .keys()),   { "default": "NONE" }),
                "filmstock":        (["NONE"] + list(self.FILMSTOCK         .keys()),   { "default": "NONE" }),
                "color":            (["NONE"] + list(self.COLOR             .keys()),   { "default": "NONE" }),
                "texture":          (["NONE"] + list(self.TEXTURE           .keys()),   { "default": "NONE" }),
                **gemini_api_parameters(),
                **self.extra_params,
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHACAM", "STRING")
    RETURN_NAMES = ("camera", "markdown")
    FUNCTION = "artha_main"

    def artha_main(self, type, context, shot, lenses, angles, motion, focal_length,depth_of_field, aspect_ratio,sensor_type, camera_style, framing, camera_models, filmstock, color, texture, api_key, model, max_tokens, temperature, randomize, use_image, image, **kwargs):
        
        response = ""
            
        markdown = self.markdown
       
        if image and use_image :
                                
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response, markdown)
            
            text_prompt = "Describe the camera settings in detail."
            
            system_instruction = load_agent("camera")
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:           
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                        
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, markdown)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, markdown)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, markdown)
                                          
        else:
            
            cam_dict = {}
            
            if randomize:
                cam_dict['TYPE']            = random.choice([key for key in self.TYPE           .keys()      if key != 'NONE'])
                cam_dict['CONTEXT']         = random.choice([key for key in self.CONTEXT        .keys()      if key != 'NONE'])
                cam_dict['SHOT']            = random.choice([key for key in self.SHOT           .keys()      if key != 'NONE'])
                cam_dict['LENSES']          = random.choice([key for key in self.LENSES         .keys()      if key != 'NONE'])
                cam_dict['ANGLES']          = random.choice([key for key in self.ANGLES         .keys()      if key != 'NONE'])
                cam_dict['MOTION']          = random.choice([key for key in self.MOTION         .keys()      if key != 'NONE'])
                cam_dict['FOCAL_LENGTH']    = random.choice([key for key in self.FOCAL_LENGTH   .keys()      if key != 'NONE'])
                cam_dict['DEPTH_OF_FIELD']  = random.choice([key for key in self.DEPTH_OF_FIELD .keys()      if key != 'NONE'])
                cam_dict['ASPECT_RATIO']    = random.choice([key for key in self.ASPECT_RATIO   .keys()      if key != 'NONE'])
                cam_dict['SENSOR_TYPE']     = random.choice([key for key in self.SENSOR_TYPE    .keys()      if key != 'NONE'])
                cam_dict['CAMERA_STYLE']    = random.choice([key for key in self.CAMERA_STYLE   .keys()      if key != 'NONE'])
                cam_dict['FRAMING']         = random.choice([key for key in self.FRAMING        .keys()      if key != 'NONE'])
                cam_dict['CAMERA_MODELS']   = random.choice([key for key in self.CAMERA_MODELS  .keys()      if key != 'NONE'])
                cam_dict['FILMSTOCK']       = random.choice([key for key in self.FILMSTOCK      .keys()      if key != 'NONE'])
                cam_dict['COLOR']           = random.choice([key for key in self.COLOR          .keys()      if key != 'NONE'])
                cam_dict['TEXTURE']         = random.choice([key for key in self.TEXTURE        .keys()      if key != 'NONE'])
                        
            else:
    
                cam_dict['TYPE']            = type
                cam_dict['CONTEXT']         = context
                cam_dict['SHOT']            = shot     
                cam_dict['LENSES']          = lenses          
                cam_dict['ANGLES']          = angles          
                cam_dict['MOTION']          = motion          
                cam_dict['FOCAL_LENGTH']    = focal_length    
                cam_dict['DEPTH_OF_FIELD']  = depth_of_field  
                cam_dict['ASPECT_RATIO']    = aspect_ratio    
                cam_dict['SENSOR_TYPE']     = sensor_type     
                cam_dict['CAMERA_STYLE']    = camera_style    
                cam_dict['FRAMING']         = framing         
                cam_dict['CAMERA_MODELS']   = camera_models   
                cam_dict['FILMSTOCK']       = filmstock                 
                cam_dict['COLOR']           = color           
                cam_dict['TEXTURE']         = texture                         
                   
            return (cam_dict, markdown)
               
################################################# 

class GeminiLight:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Light settings for image or video prompting."
    DESCRIPTION += "If use image is selected the setting are fetched "
    DESCRIPTION += "from the uploaded image by Gemini Vision. "
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")
    
    with open(json_path, 'r') as file:
        
        light = json.load(file)
        
    markdown = "# LIGHT OPTIONS"
    markdown += "\n\n"
    
    for k, v in light['LIGHT'].items():
        
        markdown += "## " + k
        markdown += "\n"
        
        if isinstance(v, dict):

            for kk, vv in v.items():
                markdown += "- " + kk + ": " + str(vv)
                markdown += "\n"
        else:
 
            markdown += "### " + str(v)
            markdown += "\n"
        
        markdown += "\n"
        
    LIGHT_SOURCE    = light['LIGHT']['SOURCE']
    LIGHT_QUALITY   = light['LIGHT']['QUALITY']
    LIGHT_DIRECTION = light['LIGHT']['DIRECTION']
    LIGHT_TIMEOFDAY = light['LIGHT']['TIMEOFDAY']
    LIGHT_PHENOMENA = light['LIGHT']['PHENOMENA']
    
    extra_params = {
        "randomize": ("BOOLEAN", {"default": False}),
        "use_image": ("BOOLEAN", {"default": False}),
    }
        
    @classmethod
    def INPUT_TYPES(self):
        
        comfyui_input_dir = folder_paths.get_input_directory()
        image_file_list = [f for f in os.listdir(comfyui_input_dir) if os.path.isfile(os.path.join(comfyui_input_dir, f))]
        
        return {
            "required": {
                "source":       (["NONE"] + list(self.LIGHT_SOURCE      .keys()),   { "default": "NONE" }),
                "quality":      (["NONE"] + list(self.LIGHT_QUALITY     .keys()),   { "default": "NONE" }),
                "direction":    (["NONE"] + list(self.LIGHT_DIRECTION   .keys()),   { "default": "NONE" }),
                "timeofday":    (["NONE"] + list(self.LIGHT_TIMEOFDAY   .keys()),   { "default": "NONE" }),
                "phenomena":    (["NONE"] + list(self.LIGHT_PHENOMENA   .keys()),   { "default": "NONE" }),
                **gemini_api_parameters(),
                **self.extra_params,
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHALIGHT", "STRING")
    RETURN_NAMES = ("light", "markdown")
    FUNCTION = "artha_main"

    def artha_main(self, source, quality, direction, timeofday, phenomena, api_key, model, max_tokens, temperature, randomize, use_image, image, **kwargs):
    
        response = ""
            
        markdown = self.markdown
        
        if image and use_image :
                     
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response, markdown)
            
            text_prompt = "Describe the light settings in detail."
            
            system_instruction = load_agent("light")
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:           
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                        
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, markdown)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, markdown)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, markdown)
                       
        else:
            
            light_dict = {}
            
            if randomize:
                
                light_dict['LIGHT_SOURCE']      = random.choice([key for key in self.LIGHT_SOURCE     .keys()      if key != 'NONE'])
                light_dict['LIGHT_QUALITY']     = random.choice([key for key in self.LIGHT_QUALITY    .keys()      if key != 'NONE'])
                light_dict['LIGHT_DIRECTION']   = random.choice([key for key in self.LIGHT_DIRECTION  .keys()      if key != 'NONE'])
                light_dict['LIGHT_TIMEOFDAY']   = random.choice([key for key in self.LIGHT_TIMEOFDAY  .keys()      if key != 'NONE'])
                light_dict['LIGHT_PHENOMENA']   = random.choice([key for key in self.LIGHT_PHENOMENA  .keys()      if key != 'NONE'])
                
            else:
    
                light_dict['LIGHT_SOURCE']      = source   
                light_dict['LIGHT_QUALITY']     = quality 
                light_dict['LIGHT_DIRECTION']   = direction 
                light_dict['LIGHT_TIMEOFDAY']   = timeofday
                light_dict['LIGHT_PHENOMENA']   = phenomena          
                   
            return (light_dict, markdown) 
               
################################################# 

class GeminiStyle:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Style settings for image or video prompting."
    DESCRIPTION += "If use image is selected the setting are fetched "
    DESCRIPTION += "from the uploaded image by Gemini Vision. "
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")
    
    with open(json_path, 'r') as file:
        
        style = json.load(file)
        
    markdown = "# STYLE OPTIONS"
    markdown += "\n\n"
    
    for k, v in style['STYLE'].items():
        
        markdown += "## " + k
        markdown += "\n"
        
        if isinstance(v, dict):

            for kk, vv in v.items():
                markdown += "- " + kk + ": " + str(vv)
                markdown += "\n"
        else:
 
            markdown += "### " + str(v)
            markdown += "\n"
        
        markdown += "\n"        
        
    STYLE_TRADITIONAL   = style['STYLE']['TRADITIONAL']
    STYLE_MODERN        = style['STYLE']['MODERN']
    STYLE_PHOTOGRAPHIC  = style['STYLE']['PHOTOGRAPHIC']
    STYLE_NAMED         = style['STYLE']['NAMED']
    STYLE_INSPIRED      = style['STYLE']['INSPIRED']
   
    extra_params = {
        "use_image": ("BOOLEAN", {"default": False}),
    }
        
    @classmethod
    def INPUT_TYPES(self):
        
        comfyui_input_dir = folder_paths.get_input_directory()
        image_file_list = [f for f in os.listdir(comfyui_input_dir) if os.path.isfile(os.path.join(comfyui_input_dir, f))]
        
        return {
            "required": {
                "traditional":  (["NONE"] + list(self.STYLE_TRADITIONAL    .keys()),   { "default": "NONE" }),
                "modern":       (["NONE"] + list(self.STYLE_MODERN         .keys()),   { "default": "NONE" }),
                "photographic": (["NONE"] + list(self.STYLE_PHOTOGRAPHIC   .keys()),   { "default": "NONE" }),
                "named":        (["NONE"] + list(self.STYLE_NAMED          .keys()),   { "default": "NONE" }),
                "inspired":     (["NONE"] + list(self.STYLE_INSPIRED       .keys()),   { "default": "NONE" }),
                **gemini_api_parameters(),
                **self.extra_params,
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHASTYLE", "STRING")
    RETURN_NAMES = ("style", "markdown")
    FUNCTION = "artha_main"

    def artha_main(self, traditional, modern, photographic, named, inspired, api_key, model, max_tokens, temperature, use_image, image, **kwargs):
    
        response = ""
        
        markdown = self.markdown
        
        if image and use_image :
                           
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response, markdown)
            
            text_prompt = "Describe the style in detail."
            
            system_instruction = load_agent("style")
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
                try:           
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                        
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, markdown)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, markdown)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, markdown)
                       
        else:
            
            style_dict = {}
      
            style_dict['STYLE_TRADITIONAL ']    = traditional
            style_dict['STYLE_MODERN']          = modern     
            style_dict['STYLE_PHOTOGRAPHIC']    = photographic
            style_dict['STYLE_NAMED']           = named     
            style_dict['STYLE_INSPIRED']        = inspired          
                   
            return (style_dict, markdown)
                         
#################################################
####################DISPLAY######################
#################################################  

class GeminiResponse:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    
    DESCRIPTION = "Gemini Response Node's objective is to display other Gemini nodes's outputs."
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "response": ("STRING", {"forceInput": True}),
                "text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    OUTPUT_NODE = True

    def artha_main(self, response, text):
        
        response = str(response)
        response = response.replace("*", "")
        response = response.replace("#", "")
       
        return {
        "ui": {"response": [response]}, 
        "result": (response,)
        }
        
#################################################

class GeminiMarkdown:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    
    DESCRIPTION = "Displays markdown text."
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "response": ("STRING", {"forceInput": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "artha_main"
    OUTPUT_NODE = True

    def artha_main(self, response):
        
        response = str(response)
       
        return {
        "ui": {"response": [response]}, 
        "result": (response,)
        }
                       
#################################################

class GeminiInstruct:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    
    DESCRIPTION = "Gemini Instruct Node's objective is to provide agent instructions for other Gemini Nodes's system instruction input slots."
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": ("STRING", {
                    "multiline": True,
                    "default": "You are an intelligent ai assistant."
                }),
                "task": ("STRING", {
                    "multiline": True,
                    "default": "Your task is to..."
                }),
                "instructions": ("STRING", {
                    "multiline": True,
                    "default": ""
                })
            },
        }
    
    RETURN_TYPES = ("ARTHAINSTRUCT",)
    RETURN_NAMES = ("system_instruction",)
    FUNCTION = "artha_main"

    def artha_main(self, role, task, instructions):
        
        role = role.replace('"', '')
        task = task.replace('"', '')
        instructions = instructions.replace('"', '')
        
        system_instruction = "Role: " + str(role) + "\n\n" + "Task: " + str(task) + "\n\n" + str(instructions)
       
        return (system_instruction,)
                       
#################################################          

# Required mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Gemini Question":  GeminiQuestion,
    "Gemini Operation": GeminiOperation,
    "Gemini Translate": GeminiTranslate,
    "Gemini Imagen":    GeminiImagen,
    "Gemini Speech":    GeminiSpeech,
    "Gemini Vision":    GeminiVision,
    "Gemini Motion":    GeminiMotion,
    "Gemini Prompter":  GeminiPrompter,
    "Gemini Condense":  GeminiCondense,
    "Gemini Portrait":  GeminiPortrait,
    "Gemini Face":      GeminiFace,
    "Gemini Body":      GeminiBody,
    "Gemini Form":      GeminiForm,
    "Gemini Cloth":     GeminiCloth,
    "Gemini Makeup":    GeminiMakeup,    
    "Gemini Backdrop":  GeminiBackdrop,
    "Gemini Compose":   GeminiCompose,
    "Gemini Subject":   GeminiSubject,
    "Gemini Scenery":   GeminiScenery,
    "Gemini Camera":    GeminiCamera,
    "Gemini Light":     GeminiLight,
    "Gemini Style":     GeminiStyle,
    "Gemini Response":  GeminiResponse,
    "Gemini Markdown":  GeminiMarkdown,
    "Gemini Instruct":  GeminiInstruct    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini Question":  node_prefix() + " Gemini Question",
    "Gemini Operation": node_prefix() + " Gemini Operation",
    "Gemini Translate": node_prefix() + " Gemini Translate",
    "Gemini Imagen":    node_prefix() + " Gemini Imagen",
    "Gemini Speech":    node_prefix() + " Gemini Speech",
    "Gemini Vision":    node_prefix() + " Gemini Vision",
    "Gemini Motion":    node_prefix() + " Gemini Motion",
    "Gemini Prompter":  node_prefix() + " Gemini Prompter", 
    "Gemini Condense":  node_prefix() + " Gemini Condense", 
    "Gemini Portrait":  node_prefix() + " Gemini Portrait",
    "Gemini Face":      node_prefix() + " Gemini Face",
    "Gemini Body":      node_prefix() + " Gemini Body",
    "Gemini Form":      node_prefix() + " Gemini Form",
    "Gemini Cloth":     node_prefix() + " Gemini Cloth",
    "Gemini Makeup":    node_prefix() + " Gemini Makeup",
    "Gemini Backdrop":  node_prefix() + " Gemini Backdrop",
    "Gemini Compose":   node_prefix() + " Gemini Compose",
    "Gemini Subject":   node_prefix() + " Gemini Subject",
    "Gemini Scenery":   node_prefix() + " Gemini Scenery",
    "Gemini Camera":    node_prefix() + " Gemini Camera",
    "Gemini Light":     node_prefix() + " Gemini Light",
    "Gemini Style":     node_prefix() + " Gemini Style",
    "Gemini Response":  node_prefix() + " Gemini Response",
    "Gemini Markdown":  node_prefix() + " Gemini Markdown",
    "Gemini Instruct":  node_prefix() + " Gemini Instruct"    
}