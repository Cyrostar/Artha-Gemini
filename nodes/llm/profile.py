import os
import json
import random

from PIL import Image
import folder_paths

from ...core.node import node_path, node_prefix, main_cetegory
from ...core.api import load_api_key
from ...core.llm import call_gemini_api
from ...core.llm import load_agent

class GeminiPortrait:
     
    CATEGORY = main_cetegory() + "/LLM"
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

        comfyui_input_dir = folder_paths.get_input_directory() 
        os.makedirs(os.path.join(comfyui_input_dir, "artha"), exist_ok=True) 
        os.makedirs(os.path.join(comfyui_input_dir, "artha", "portrait"), exist_ok=True)
        os.makedirs(os.path.join(comfyui_input_dir, "artha", "face"), exist_ok=True)
        os.makedirs(os.path.join(comfyui_input_dir, "artha", "body"), exist_ok=True) 
        os.makedirs(os.path.join(comfyui_input_dir, "artha", "form"), exist_ok=True)
        os.makedirs(os.path.join(comfyui_input_dir, "artha", "cloth"), exist_ok=True)
        os.makedirs(os.path.join(comfyui_input_dir, "artha", "makeup"), exist_ok=True)         
 
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
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API key will be visible in plain text. Consider adding your api to the api.json located inside this custom node folder."
                }),
                "model": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"], {
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,}),
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
    
    def artha_main(self, text_prompt, identity, framing, api_key, model, max_tokens, temperature, seed, use_image, reconstruct, image, **kwargs):
        
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
            
            if isinstance(face, dict):
            
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
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
            
            try:
                
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
                
                return (response, property_list)
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (error_msg,)
                
                
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
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
            
            try:
                
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
                
                return (response, property_list)                
                
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (error_msg,)
            
        else:
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            system_instruction += "\n"
                       
            system_instruction += "If property list is empty return an empty response. \n\n"
               
            try:
                
                pil_image = None
                
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
        
                return (response, property_list)                 
            
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (error_msg,)
                
#################################################

class GeminiFace:
    
    CATEGORY = main_cetegory() + "/LLM"
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
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "randomize":            ("BOOLEAN", {"default": False}),
                "use_image":            ("BOOLEAN", {"default": False}),
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHAFACE",)
    RETURN_NAMES = ("face",)
    FUNCTION = "artha_main"

    def artha_main(self, head_type, hair_color, hair_length, hair_style_fem, hair_style_mas, face_appeal, face_age, face_shape, face_eyebrow_type, face_eyebrow_shape, face_eye_type, face_eye_size, face_eye_color, face_nose_type, face_lip_type, face_lip_color, face_ear_type, face_cheek_type, face_chin_type, api_key, model, max_tokens, temperature, randomize, use_image, image):
              
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
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                    
            system_instruction = load_agent("face")
                    
            try:           
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
                response = response.upper()
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
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
    
    CATEGORY = main_cetegory() + "/LLM"
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
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "randomize":            ("BOOLEAN", {"default": False}),
                "use_image":            ("BOOLEAN", {"default": False}),
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHABODY",)
    RETURN_NAMES = ("body",)
    FUNCTION = "artha_main"

    def artha_main(self, body_type, body_height, body_weight, body_build, body_frame, body_shoulder, body_chest, body_breasts, body_torso, body_waist, body_hip, body_legs, body_skin_tone, body_posture, api_key, model, max_tokens, temperature, randomize, use_image, image):
    
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
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                    
            system_instruction = load_agent("body")
                    
            try:           
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
                response = response.upper()
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
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
    
    CATEGORY = main_cetegory() + "/LLM"
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
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "use_image": ("BOOLEAN", {"default": False}),
                "image": (sorted(image_file_list), {"image_upload": True}),
            }
        }
    
    RETURN_TYPES = ("ARTHAFORM",)
    RETURN_NAMES = ("form",)
    FUNCTION = "artha_main"

    def artha_main(self, chest, shoulders, arms, biceps, triceps, forearms, abs, core, obliques, back, lats, traps, legs, quadriceps, hamstrings, calves, glutes, api_key, model, max_tokens, temperature, use_image, image):
        
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
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = Image.open(image_path).convert("RGB")
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                    
            system_instruction = load_agent("form")
                    
            try:           
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
    
    CATEGORY = main_cetegory() + "/LLM"
    DESCRIPTION = "Describes clothing of the subject."
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
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
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "image": (sorted(files), {"image_upload": True}),                
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
           
    RETURN_TYPES = ("ARTHACLOTH",)
    RETURN_NAMES = ("cloth",)
    FUNCTION = "artha_main"

    def artha_main(self, image, api_key, model, max_tokens, temperature, extra_pnginfo=None):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Identify the clothes and list each one."
        
        image_path = folder_paths.get_annotated_filepath(image)
        
        try:
            
            pil_image = Image.open(image_path).convert("RGB")
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
                 
        system_instruction = load_agent("cloth")
                
        try:           
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
            response = response.upper()
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
#################################################

class GeminiMakeup:
    
    CATEGORY = main_cetegory() + "/LLM"
    DESCRIPTION = "Describes make-up of the subject."
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
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
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "image": (sorted(files), {"image_upload": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
           
    RETURN_TYPES = ("ARTHAMAKEUP",)
    RETURN_NAMES = ("makeup",)
    FUNCTION = "artha_main"

    def artha_main(self, image, api_key, model, max_tokens, temperature, extra_pnginfo=None):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Identify the make-up of the face and list each element."
        
        image_path = folder_paths.get_annotated_filepath(image)
        
        try:
            
            pil_image = Image.open(image_path).convert("RGB")
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
                 
        system_instruction = load_agent("makeup")
                
        try:           
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
            response = response.upper()
            
            return (response,)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
#################################################

class GeminiBackdrop:
    
    CATEGORY = main_cetegory() + "/LLM"
    DESCRIPTION = "Describes the background of an image."
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
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
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "A temperature of 0 means only the most likely tokens are selected, and there's no randomness. Conversely, a high temperature injects a high degree of randomness into the tokens selected by the model, leading to more unexpected, surprising model responses."
                }),
                "image": (sorted(files), {"image_upload": True}),                
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
           
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("backdrop",)
    FUNCTION = "artha_main"

    def artha_main(self, image, api_key, model, max_tokens, temperature, extra_pnginfo=None):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Describe the background."
        
        image_path = folder_paths.get_annotated_filepath(image)
        
        try:
            
            pil_image = Image.open(image_path).convert("RGB")
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
                 
        system_instruction = load_agent("backdrop")
                
        try:           
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
            return (response,)
               
#################################################        

# Required mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Gemini Portrait": GeminiPortrait,
    "Gemini Face": GeminiFace,
    "Gemini Body": GeminiBody,
    "Gemini Form": GeminiForm,
    "Gemini Cloth": GeminiCloth,
    "Gemini Makeup": GeminiMakeup,    
    "Gemini Backdrop": GeminiBackdrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini Portrait": node_prefix() + " Gemini Portrait",
    "Gemini Face": node_prefix() + " Gemini Face",
    "Gemini Body": node_prefix() + " Gemini Body",
    "Gemini Form": node_prefix() + " Gemini Form",
    "Gemini Cloth": node_prefix() + " Gemini Cloth",
    "Gemini Makeup": node_prefix() + " Gemini Makeup",
    "Gemini Backdrop": node_prefix() + " Gemini Backdrop",
}