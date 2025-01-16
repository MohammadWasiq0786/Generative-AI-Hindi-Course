import gradio as gr
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login

# Hugging Face Login
# HUGGINGFACEHUB_API_TOKEN = "your_huggingface_api_key"  # Replace with your Hugging Face API key

from google.colab import userdata
HUGGINGFACEHUB_API_TOKEN= userdata.get('HUGGING_FACE_API_KEY')
login(token=HUGGINGFACEHUB_API_TOKEN)

# Hugging Face Model Configuration
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
chat = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=2048,
    temperature=0.7,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# Template for blog generation
template = """
    As an experienced startup and venture capital writer, 
    generate a 1000-word blog post about {topic}.
    Your response should be in a point-wise manner and give in a proper question-answer format as well.

    Your response should be in this format:
    First, print the blog post.
    Then, sum the total number of words in it and print the result like this: This post has 1000 words.
"""

# Define the PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)

# Function to generate blog post
def generate_blog_post(topic):
    if not topic.strip():
        return "Please enter a topic to generate a blog post."

    formatted_prompt = prompt.format(topic=topic)
    response = chat(formatted_prompt)
    return response

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("""
        # Blog Post Generator by Mohammad Wasiq
        Generate a blog post in a structured format using LLM.
    """)

    topic_input = gr.Textbox(label="Enter Topic:", placeholder="Type Your Blog Topic Here...")
    output = gr.Textbox(label="Generated Blog Post:", lines=50, interactive=False)

    generate_button = gr.Button("Generate Blog Post")
    generate_button.click(generate_blog_post, inputs=[topic_input], outputs=[output])

# Launch the Gradio interface
demo.launch()