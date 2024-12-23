import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_project'] = os.getenv("LANGCHAIN_project")

# Streamlit app setup
st.title("llama2")

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    "System: Namaskar Lauda JI\nUser: Question: {question}"
)

# Input field for user question
input_text = st.text_input("KYU RE LAUDE WAPAS AAGYA TU ")

# LLM setup (ensure Llama2 is accessible via Ollama API)
try:
    # Use Ollama correctly with the required parameters
    llm = Ollama(model="llama2:latest")  # Pass 'model' as a keyword argument
    output_parser = StrOutputParser()

    # Process input and display the result
    if input_text:
        try:
            # Combine prompt, LLM, and parser
            chain = prompt | llm | output_parser
            response = chain.invoke({"question": input_text})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
