import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langchain.output_parsers import StructuredOutputParser,ResponseSchema



wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1024)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

loader = PyPDFLoader(r"C:\Users\Garv Khurana\OneDrive\Desktop\Langchain\advancerag\attention.pdf")
document = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(document)
db = FAISS.from_documents(chunks, OllamaEmbeddings(model="llama3"))
retriever = db.as_retriever()

pdf_retriever_tool = create_retriever_tool(retriever, name="chatbot", description="you are a helpful assistant")

arxiv_wrapper = ArxivAPIWrapper(top_k_results=2)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tools = [wiki_tool, arxiv_tool, pdf_retriever_tool]

llm = Ollama(model="llama3")

prompt_template = """
You are a helpful assistant. Give me an answer related to user query: {query}
Agent scratchpad: {agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(
        name="source",
        description="source used to answer the user's question, should be a website.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

agent=create_openai_tools_agent(llm,tools,prompt)


agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, output_parser=output_parser)


st.title("Langchain Assistant")

st.write("Ask a question and get an answer using the Langchain agent.")

query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Processing..."):
            response = agent_executor.invoke({"query": query})
            st.write(response["output"])
    else:
        st.warning("Please enter a query.")
        
        
        
        
        
        
        
