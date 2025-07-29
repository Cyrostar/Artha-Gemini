from ...core.node import node_prefix, main_cetegory

#################################################  

class GeminiResponse:
    
    CATEGORY = main_cetegory() + "/LLM"
    
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
    
    CATEGORY = main_cetegory() + "/LLM"
    
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
    
    CATEGORY = main_cetegory() + "/LLM"
    
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
    "Gemini Response":  GeminiResponse,
    "Gemini Markdown":  GeminiMarkdown,
    "Gemini Instruct":  GeminiInstruct
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini Response":  node_prefix() + " Gemini Response",
    "Gemini Markdown":  node_prefix() + " Gemini Markdown",
    "Gemini Instruct":  node_prefix() + " Gemini Instruct"
}