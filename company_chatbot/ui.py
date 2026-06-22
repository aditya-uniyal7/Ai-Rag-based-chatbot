import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_ollama import ChatOllama

# 1. Setup (Cache taaki baar-baar embed na ho)
@st.cache_resource
def load_data():
    loader = TextLoader("policies.txt", encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50, separators=["Q:"]).split_documents(loader.load())
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    return FAISS.from_documents(chunks, embeddings)

vector_store = load_data()
llm = ChatOllama(model="qwen3:0.6b")

# 2. UI
st.set_page_config(page_title="TechShop Support", page_icon="💻")
st.title("💻 TechShop Support Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Chat Logic
if prompt := st.chat_input("Apna sawaal puchein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval
    docs = vector_store.similarity_search(prompt, k=1)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # Prompt
    temp = f"Aap TechShop assistant hain. Context: {context_text} \n   Question: {prompt} \n Assistant:"
    
    # Response
    with st.chat_message("assistant"):
        response = llm.invoke(temp)
        answer = response.content if hasattr(response, 'content') else str(response)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})