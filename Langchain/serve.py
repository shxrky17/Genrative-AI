from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


groq_api_key=os.getenv("GROQ_API_KEY")
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


genric_template="tell car of this brand"

prompt=ChatPromptTemplate.from_messages(
    [("system",genric_template),("user","{text}")]
)

parser=StrOutputParser()

chain=prompt|model|parser
app=FastAPI(title="Langchain Server",version="1.0",description="ji")
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
    