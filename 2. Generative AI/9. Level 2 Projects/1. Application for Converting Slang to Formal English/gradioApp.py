import gradio as gr
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login

# Login to Hugging Face
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_api_key"  # Replace with your API key
login(token=HUGGINGFACEHUB_API_TOKEN)

# Hugging Face Model Configuration
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
chat = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.7,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# Template for rewriting text
template = """
    Below is a draft text that may be poorly worded.
    Your goal is to:
    - Properly redact the draft text
    - Convert the draft text to a specified tone
    - Convert the draft text to a specified dialect

    Here are some examples of different Tones:
    - Formal: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - Informal: Hey everyone, it's been a wild week! We've got some exciting news to share - Sam Altman is back at OpenAI, taking up the role of chief executive. After a bunch of intense talks, debates, and convincing, Altman is making his triumphant return to the AI startup he co-founded.  

    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, \ cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flag, rubbish, biscuit, green fingers, car park, trousers, windscreen

    Example Sentences from each dialect:
    - American: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - British: On Wednesday, OpenAI, the esteemed artificial intelligence start-up, announced that Sam Altman would be returning as its Chief Executive Officer. This decisive move follows five days of deliberation, discourse and persuasion, after Altman's abrupt departure from the company which he had co-established.

    Please start the redaction with a warm introduction. Add the introduction if you need to.
    
    Below is the draft text, tone, and dialect:
    DRAFT: {draft}
    TONE: {tone}
    DIALECT: {dialect}

    YOUR {dialect} RESPONSE:
"""

# Define the PromptTemplate
prompt = PromptTemplate(
    input_variables=["tone", "dialect", "draft"],
    template=template,
)

# Function for text rewriting
def rewrite_text(draft, tone, dialect):
    if len(draft.split(" ")) > 700:
        return "Please enter a shorter text. The maximum length is 700 words."

    formatted_prompt = prompt.format(tone=tone, dialect=dialect, draft=draft)
    response = chat(formatted_prompt)
    return response

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("""
        # Re-write Your Text
        This app allows you to rewrite text in different tones and dialects using an open-source Hugging Face model.
    """)

    with gr.Row():
        tone = gr.Dropdown(["Formal", "Informal"], label="Select Tone", value="Formal")
        dialect = gr.Dropdown(["American", "British"], label="Select Dialect", value="American")

    draft_input = gr.Textbox(label="Enter your draft text", placeholder="Your text here...", lines=10)
    output_text = gr.Textbox(label="Rewritten Text", lines=10, interactive=False)

    rewrite_button = gr.Button("Rewrite Text")
    rewrite_button.click(rewrite_text, inputs=[draft_input, tone, dialect], outputs=output_text)

# Launch the Gradio interface
demo.launch()
