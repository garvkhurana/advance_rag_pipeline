import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain.indexes import FAISSIndex
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


ollama_model = Ollama(model_name="llama3")
embeddings = OllamaEmbeddings(model="llama3")


loader = PyPDFLoader('attention.pdf')
docs =loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = splitter.split_documents(docs)


index = FAISSIndex(dim=128)


for doc in chunked_docs:
    doc_embedding = embeddings.get_embedding(doc)
    index.add(doc_embedding)


prompt_template = ChatPromptTemplate.from_messages(
    prompt="Answer the following question based on the provided context:\n\n{question}\n\nContext:\n{context}",
    output_key="answer"
)

# Define a function to answer user queries
def answer_query(user_query):
    query_embedding = embeddings.get_embedding(user_query)
    nearest_neighbors = index.search(query_embedding, k=5)
    context_docs = [chunked_docs[i] for i in nearest_neighbors]
    context = "\n\n".join(context_docs)
    prompt = prompt_template.format(question=user_query, context=context)
    answer = RetrievalQA(ollama_model, context_docs).answer(prompt)
    return answer


st.title("Doctor Bot")
st.write("Ask me a question!")


user_query = st.text_input("")

# Answer button
if st.button("Get Answer"):
    answer = answer_query(user_query)
    st.write(answer)