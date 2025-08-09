import os
import folder_paths

ncm = {}
ndm = {}

from .llm.gemini import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

for k, v in NODE_CLASS_MAPPINGS.items():
    
    ncm[k] = v
    
for k, v in NODE_DISPLAY_NAME_MAPPINGS.items():
    
    ndm[k] = v

NODE_CLASS_MAPPINGS = ncm
NODE_DISPLAY_NAME_MAPPINGS = ndm

comfyui_input_dir = folder_paths.get_input_directory() 
os.makedirs(os.path.join(comfyui_input_dir, "artha"), exist_ok=True)
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini"), exist_ok=True)  
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "portrait"), exist_ok=True)
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "face"), exist_ok=True)
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "body"), exist_ok=True) 
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "form"), exist_ok=True)
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "cloth"), exist_ok=True)
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "makeup"), exist_ok=True) 
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "scene"), exist_ok=True)
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "camera"), exist_ok=True)
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "light"), exist_ok=True) 
os.makedirs(os.path.join(comfyui_input_dir, "artha", "gemini", "style"), exist_ok=True)