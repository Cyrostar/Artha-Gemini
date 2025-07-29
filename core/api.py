import os
import json
from ..core.node import node_path

def load_api_key(provider):
    
    node_dir = node_path()
    json_path = os.path.join(node_dir, "api.json")
        
    try:
        
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        return json_file.get(provider + '_api_key')
        
    except Exception as e:
        
        print(f"Error loading json file: {e}")
        return None