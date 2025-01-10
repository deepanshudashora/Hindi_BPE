import gradio as gr
from train import HindiBPETokenizer
import json
import os

loaded_tokenizer = HindiBPETokenizer.load_tokenizer("hindi_tokenizer_v4")

# Example texts
EXAMPLES = [
    ["मैं आज बहुत खुश हूं"],
    ["भारत एक विशाल देश है"],
    ["मुझे हिंदी भाषा बहुत पसंद है"],
    ["आज का दिन बहुत सुंदर है"],
]

def process_text(input_text):
    try:
        encoded = loaded_tokenizer.encode(input_text)
        decoded = loaded_tokenizer.decode(encoded)
        
        tokens_str = str(encoded)
        formatted_output = f"""### Tokenization Results:

**Encoded Tokens:**
```
{tokens_str}
```

**Decoded Text:**
```
{decoded}
```

**Number of Tokens:**
```
{len(encoded)}
```"""
        return formatted_output
    except Exception as e:
        return f"""### Error Occurred:
```
{str(e)}
```"""

# Custom CSS
custom_css = """
.container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}

.title {
    text-align: center;
    color: #f0f0f0; /* Lighter text for dark backgrounds */
    margin-bottom: 20px;
}

.description {
    text-align: center;
    color: #e2e2e2;
    margin-bottom: 30px;
}

.input-box {
    border: 1px solid #4a5568;
    border-radius: 8px;
    padding: 15px;
    background-color: #1a202c; /* Dark background for input */
    color: #fff;
}

/* Updated to match a dark theme */
.output-box {
    border: 1px solid #4a5568;
    border-radius: 8px;
    padding: 15px;
    background-color: #1a202c; /* Dark background for output */
    color: #fff;
}

.output-box h3 {
    color: #fff;
    margin-top: 0;
}

.output-box strong {
    color: #a0aec0;
}

/* Adjust pre block styling for dark background */
.output-box pre {
    background-color: #2d3748;
    padding: 12px;
    border-radius: 6px;
    margin: 8px 0;
    color: #e2e2e2;
    font-size: 1.1em;
    font-weight: 500;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.examples-heading {
    margin-top: 20px;
    margin-bottom: 10px;
    color: #fff;
    font-size: 1.2em;
}

"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <div class="title">
            <h1>🔤 Hindi BPE Tokenizer Demo</h1>
        </div>
        <div class="description">
            <p>This demo showcases a BPE (Byte Pair Encoding) tokenizer trained specifically for Hindi text. 
            Enter Hindi text in the input box to see how it gets tokenized.</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="यहां हिंदी टेक्स्ट दर्ज करें...",
                lines=5,
                elem_classes=["input-box"]
            )
            submit_btn = gr.Button("Tokenize", variant="primary")
        
        with gr.Column(scale=1):
            output = gr.Markdown(
                label="Results",
                elem_classes=["output-box"]
            )
    
    gr.Markdown("### Examples", elem_classes=["examples-heading"])
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=input_text,
        outputs=output,
        fn=process_text,
        cache_examples=True
    )
    
    # Add event handler
    submit_btn.click(
        fn=process_text,
        inputs=[input_text],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch(share=True)  # Added share=True to create a public link

