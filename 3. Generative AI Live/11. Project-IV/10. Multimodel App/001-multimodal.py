import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chain_gpt_35 = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
chain_gpt_4_vision = ChatOpenAI(model="gpt-4o", max_tokens=1024)

from typing import Any
import os
from unstructured.partition.pdf import partition_pdf
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

input_path = os.getcwd()
output_path = os.path.join(os.getcwd(), "figures")

# Get elements
raw_pdf_elements = partition_pdf(
    filename=os.path.join(input_path, "startupai-financial-report-v2.pdf"),
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=output_path,
)

import base64

text_elements = []
table_elements = []
image_elements = []

# Function to encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# We create the text and table elements in 2 steps
# Step 1: append the entire class in the list
for element in raw_pdf_elements:
    #Text elements have CompositeElement in the string of their type name
    if 'CompositeElement' in str(type(element)):
        text_elements.append(element)
    #Table element have Table in the string of their type name
    elif 'Table' in str(type(element)):
        table_elements.append(element)

# Step 2: extract just the text, we don't want to store the raw classes
table_elements = [i.text for i in table_elements]
text_elements = [i.text for i in text_elements]

# Tables
print("\n----------\n")

print("number of table elements in the pdf file: ")

print("\n----------\n")
print(len(table_elements))

print("\n----------\n")

# Text
print("\n----------\n")

print("number of text elements in the pdf file: ")

print("\n----------\n")
print(len(text_elements))

print("\n----------\n")


for image_file in os.listdir(output_path):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(output_path, image_file)
        encoded_image = encode_image(image_path)
        image_elements.append(encoded_image)

# Images
print("\n----------\n")

print("number of image elements in the pdf file: ")

print("\n----------\n")
print(len(image_elements))

print("\n----------\n")

from langchain.schema.messages import HumanMessage, AIMessage

# Function for text summaries
def summarize_text(text_element):
    prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for table summaries
def summarize_table(table_element):
    prompt = f"Summarize the following table:\n\n{table_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for image summaries
def summarize_image(encoded_image):
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {
                "type": "text", 
                "text": "Describe the contents of this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = chain_gpt_4_vision.invoke(prompt)
    return response.content

# Processing text elements, stopping at the 2nd
text_summaries = []
for i, te in enumerate(text_elements[0:2]):
    summary = summarize_text(te)
    text_summaries.append(summary)
    print(f"{i + 1}th element of texts processed.")
    
# Processing table elements, stopping at the 1st
table_summaries = []
for i, te in enumerate(table_elements[0:1]):
    summary = summarize_table(te)
    table_summaries.append(summary)
    print(f"{i + 1}th element of tables processed.")
    
# Processing image elements, stopping at the 8th
image_summaries = []
for i, ie in enumerate(image_elements[0:8]):
    summary = summarize_image(ie)
    image_summaries.append(summary)
    print(f"{i + 1}th element of images processed.")
    
import uuid

from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma

# Initialize the Chroma vector database and docstore
vectorstorev2 = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
storev2 = InMemoryStore()
id_key = "doc_id"

# Initialize the multi-vector retriever
retrieverv2 = MultiVectorRetriever(vectorstore=vectorstorev2, docstore=storev2, id_key=id_key)

# Function to add documents to the multi-vector retriever
def add_documents_to_retriever(summaries, original_contents):
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]
    retrieverv2.vectorstore.add_documents(summary_docs)
    retrieverv2.docstore.mset(list(zip(doc_ids, original_contents)))
    
# Add text summaries
add_documents_to_retriever(text_summaries, text_elements)

# Add table summaries
add_documents_to_retriever(table_summaries, table_elements)

# Add image summaries
add_documents_to_retriever(image_summaries, image_summaries)



print("\n----------\n")

print("What do you see in the images?")

print("\n----------\n")

print(retrieverv2.invoke(
    "What do you see in the images?"
))

print("\n----------\n")

from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

template = """Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

chain = (
    {"context": retrieverv2, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print("\n----------\n")

print("What do you see on the images in the database?")

print("\n----------\n")

print(chain.invoke(
     "What do you see on the images in the database?"
))

print("\n----------\n")

print("What is the name of the company?")

print("\n----------\n")

print(chain.invoke(
     "What is the name of the company?"
))

print("\n----------\n")

print("What is the product displayed in the image?")

print("\n----------\n")

print(chain.invoke(
     "What is the product displayed in the image?"
))

print("\n----------\n")

print("How much are the total expenses of the company?")

print("\n----------\n")

print(chain.invoke(
     "How much are the total expenses of the company?"
))

print("\n----------\n")

print("What is the ROI?")

print("\n----------\n")

print(chain.invoke(
     "What is the ROI?"
))

print("\n----------\n")

print("How much did the company sell in 2023?")

print("\n----------\n")

print(chain.invoke(
     "How much did the company sell in 2023?"
))

print("\n----------\n")

print("And in 2022?")

print("\n----------\n")

print(chain.invoke(
     "And in 2022?"
))