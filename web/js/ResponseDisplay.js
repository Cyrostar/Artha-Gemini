import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Artha.LLM.Gemini.Response",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
		if (nodeData.name === "GeminiResponse") {
        
            const onNodeCreated = nodeType.prototype.onNodeCreated;
			
            nodeType.prototype.onNodeCreated = function () {
				
				onNodeCreated?.apply(this, arguments);
				
            };
			
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            
			nodeType.prototype.onConnectionsChange = async function (type, slot, connected, link, ioSlot) {
				
				onConnectionsChange?.apply(this, arguments);
				
				if (connected && link) {
                    // Store the information about the node connected to our input
					this.connectedNodeId = link.origin_id;
                    this.connectedNodeSlot = link.origin_slot;
                } else {
                    // Clear the stored info if disconnected
                    this.connectedNodeId = null;
                    this.connectedNodeSlot = null;
                }
				
			};
			
			const onExecuted = nodeType.prototype.onExecuted;
			
            nodeType.prototype.onExecuted = async function (message) {
                
				onExecuted?.apply(this, arguments);
				
				const myNodeWidgetText = this.widgets.find(w => w.name === "text");
				
				myNodeWidgetText.value = message.response[0];

			};                                        

        }
		
    }
	
});