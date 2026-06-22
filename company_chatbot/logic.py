import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

loader = TextLoader("policies.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       
    chunk_overlap=50,
    separators=["Q:"]    
)
chunks = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b") 
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("policies_embedding")


temp = """if user ask question in engish then answer him in english else higlish
Aap TechShop ke customer support assistant hain. 
Aapke paas niche diya gaya context hai. 
Context ka use karke user ke sawaal ka jawaab dein.
Agar jawaab context mein nahi hai, toh politely kahein ki "Mujhe is baare mein jaankari nahi hai, ap kripya 1800 **** pr sahayta ke lie sampark kro".

Context: {context}

User ka sawaal: {question}

Jawab (Hinglish/Hindi mein):
"""

llm = ChatOllama(model="qwen3:0.6b") 

print("\nTechShop Support Ready! (Exit ke liye 'quit' likhein)")
while True:
    query = input("\nUser: ")
    if query.lower() == 'quit':
        break
    
    docs = vector_store.similarity_search(query, k=1)
    context_text = "\n\n".join([d.page_content for d in docs])

    final_prompt = temp.format(context=context_text, question=query)

    response = llm.invoke(final_prompt)
    print(f"\nAssistant: {response}")

    if hasattr(response, 'content'):
        print(f"\nAssistant: {response.content}")
    else:
        # Agar response string hai, toh seedha print karein
        print(f"\nAssistant: {response}")


        import streamlit as st

# Page Configuration
st.set_page_config(page_title="TechShop Support", page_icon="💻")

# UI Style
st.title("💻 TechShop Support Assistant")
st.markdown("---")

# Sidebar for FAQs
with st.sidebar:
    st.header("Quick Help")
    if st.button("Return Policy"): st.write("30 days return window.")
    if st.button("Warranty Info"): st.write("1 year on Electronics.")
    st.markdown("---")
    st.info("Contact: 1800-XXX-XXXX")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Apna sawaal puchein..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response (Yahan apna Retrieval logic call karein)
    with st.chat_message("assistant"):
        # response = qa_chain.invoke(prompt) -- Aapka retrieval logic
        response = "Yeh raha aapka jawab..." # Sample response
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})