import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMMathChain, LLMChain
from langchain.utilities import WikipediaAPIWrapper

# Streamlit Page Config
st.set_page_config(page_title="Text to Math Problem Solver")

# Sidebar for API Key Input
groq_api_key = st.sidebar.text_input(label="Enter API Key", type="password")
if not groq_api_key:
    st.info("Please provide an API key")
    st.stop()

# Initialize LLM and Tools
llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)
wiki_wrapper = WikipediaAPIWrapper()
wikitool = Tool(
    name="wikipedia",
    func=wiki_wrapper.run,
    description="Tool for searching the internet and solving math problems"
)

math_chain = LLMMathChain.from_llm(llm=llm)
calci = Tool(
    name="calci",
    func=math_chain.run,
    description="Tool for solving math problems"
)

# Prompt Template
prompt = """
You are an agent tasked with solving users' mathematical questions. Arrive at the solution logically and provide a detailed explanation, displaying it pointwise.
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="reasoning",
    func=chain.run,
    description="Tool for solving math problems"
)

# Initialize Agent
assistant_agent = initialize_agent(
    tools=[wikitool, calci, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Chat State Initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot who can answer all your math problems and questions."}
    ]

# Display Chat Messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to Generate Response
def generate_res(question):
    response = assistant_agent.run({"input": question})
    return response

# User Input
question = st.text_area("Enter your question")
if st.button("Find My Answer"):
    if question:
        with st.spinner("Thinking..."):
            # Update User Message
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            
            # Generate Response
            try:
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                res = assistant_agent.run(question, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": res})
                st.chat_message("assistant").write(res)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question.")
