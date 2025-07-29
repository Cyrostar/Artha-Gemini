import os
import folder_paths
from pathlib import Path

def node_path():
    
    file = Path(__file__)
    node_name = file.parent.parent.name
    node_path = os.path.join(folder_paths.base_path, "custom_nodes", node_name)
    
    if node_path:
        
        return node_path
            
    else:
        
        return None
        
def node_prefix():
    
    prefix = "🔱"
        
    return prefix
    
def main_cetegory():
    
    name = "Artha"
        
    return name
     