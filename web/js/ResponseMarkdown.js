import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Artha.LLM.Gemini.Markdown",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
		if (nodeData.name === "Gemini Markdown") {
        
            const onNodeCreated = nodeType.prototype.onNodeCreated;
			
            nodeType.prototype.onNodeCreated = function () {
				
				onNodeCreated?.apply(this, arguments);
				
				this.div = document.createElement("div");
				this.div.style.background = "#333";
				this.div.style.color = "#fff";
				this.div.style.padding = "10px 10px 20px 10px";
				this.div.style.margin = "5px 5px 20px 5px";
				this.div.style.borderRadius = "4px";
				this.div.style.fontFamily = "'Liberation Mono', sans-serif;";
				this.div.style.fontSize = "12px";
				this.div.style.minHeight = "60px";
				this.div.style.whiteSpace = "pre-wrap";
				this.div.style.wordWrap = "break-word";
				this.div.style.display = "block";
				this.div.style.width = "100%";
				this.div.style.overflow = "auto",
				this.div.style.boxSizing = "border-box";
				
				this.div.textContent = "Display text will appear here...";
				
				this.div.classList.add("response-markdown-output");
				
				const response_markdown_style = document.createElement("style");
				
				response_markdown_style.textContent = `
					.response-markdown-output h1 {
						color: #ffcc00;
						font-size: 18px;
						margin: 0;
					}
				
					.response-markdown-output h2 {
						color: #66d9ef;
						font-size: 16px;
						margin: 0;
					}
				
					.response-markdown-output h3 {
						color: #a6e22e;
						font-size: 14px;
						margin: 0;
					}
					
					.response-markdown-output ul {
						margin-top: 0;
					}
				`;
				
				document.head.appendChild(response_markdown_style);
				
				this.addDOMWidget("Display", "html", this.div);
				
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
				
				function markdown2html(m) {
					
					let html = m;
					
					// Code block
					html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
						return `<pre><code>${code.trim()}</code></pre>`;
					});
					
					// Headers
					html = html.replace(/^###### (.*$)/gim, '<h6>$1</h6>')
					html = html.replace(/^##### (.*$)/gim, '<h5>$1</h5>')
					html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>')
					html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>')
					html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>')
					html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>')

					// Handle strikethrough
					html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');
				
					// Handle bold-italic (***text*** or ___text___)
					html = html.replace(/(\*\*\*|___)(.*?)\1/g, '<b><i>$2</i></b>');
				
					// Handle bold (process after bold-italic!)
					html = html.replace(/(\*\*|__)(.*?)\1/g, '<b>$2</b>');
				
					// Handle italic
					html = html.replace(/(\*|_)(.*?)\1/g, '<i>$2</i>');
					
					// Images: ![alt](url) → <img src="url" alt="alt" />
					html = html.replace(/!\[([^\]]*)\]\((https?:\/\/[^\s)]+)\)/g, '<img src="$2" alt="$1" />');
					
					// Links: [Google](https://www.google.com)
					html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
					
					// Email links: [myemail](mailto:someone@example.com)
					html = html.replace(/\[([^\]]+)\]\((mailto:[^)]+)\)/g, '<a href="$2">$1</a>');
					
					// Convert markdown list markers to <li>
					html = html.replace(/^\s*[\*\-]\s+(.*)$/gm, '<li>$1</li>');
	
					// Wrap consecutive <li> items in <ul>
					html = html.replace(/(<li>.*?<\/li>\s*)+/gs, match => {
						return `<ul>\n${match.trim()}\n</ul>`;
					});					
							
					return html;

				}
				
				this.div.innerHTML = markdown2html(message.response[0]);

			};                                        

        }
		
    }
	
});