import os
import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain  # Correctly import LLMChain
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")


# Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
model_id = "mistral.mistral-7b-instruct-v0:2"

llm = Bedrock(
    model_id=model_id,
    client=bedrock,
    model_kwargs={"temperature": 0.9}
)



# Function for chatbot response
def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language", "user_text"],
        template="You are a chatbot, and you are fluent in {language}. \n\n{user_text}"
    )
    
    # Pass the Bedrock LLM (not a string) to LLMChain
    chatbot_chain = LLMChain(llm=llm, prompt=prompt)
    response = chatbot_chain.run({"language": language, "user_text": user_text})
    
    return response

def main():
    st.title("Language Chatbot")
    
    # Sidebar input for language selection
    language = st.sidebar.selectbox("Choose the language:", ["English", "Spanish", "French", "German"])

    # Text input for user question
    user_text = st.sidebar.text_area("Ask your question here")

    if user_text:  # If the user has provided input text
        response = my_chatbot(language, user_text)
        st.write(response)  # Just write the response as a string

if __name__ == "__main__":
    main()
