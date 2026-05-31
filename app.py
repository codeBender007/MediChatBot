import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# LangChain & HuggingFace Core Imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Vector Store Files Path Configuration
DB_FAISS_PATH = 'vectoreStore/db_faiss'


@st.cache_resource
def get_vectorstore():
    """Vector database ko load aur cache karta hai."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    try:
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception:
        # Agar vector store na mile toh error dene ke bajaye None return karega taaki normal chat chalti rahe
        return None


@st.cache_resource
def load_model(huggingface_repo_id, hf_token):
    """HuggingFace endpoint initiate karta hai."""
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.7,  # Temperature thoda badhaya taaki general questions me model creative answer de sake
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


def format_docs(docs):
    """Retrieved chunks ko single context string me join karta hai."""
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    st.set_page_config(page_title="Medibot AI", page_icon="🤖")
    st.title("Ask Chatbot (RAG + General AI)")

    # Environment Token Check
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        st.error("Error: 'HF_TOKEN' nahi mila. Kripya apni .env file check karein.")
        st.stop()

    huggingface_repo_id = "Qwen/Qwen2.5-7B-Instruct"

    # Global Components Initialization
    try:
        vectorstore = get_vectorstore()
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        else:
            retriever = None
        llm_model = load_model(huggingface_repo_id, HF_TOKEN)
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        return

    # Persistent Session Chat History Layer
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # UI par puraani chat history render karna
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    # Chat Input Box
    prompt = st.chat_input("Ask Me Anything...")

    if prompt:
        # 1. User Message Display aur State update
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Hybrid Prompt Template (Yahan badlaav kiya hai)
        custom_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and intelligent AI assistant. 
You are provided with some context from documents. 
1. If the answer can be found in the context, prioritize using that context to answer.
2. If the context does not contain the answer, use your own general knowledge to answer the user's question accurately.
3. Keep your answers direct, clear, and helpful."""),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

        # 3. Dynamic Chain Setup (Agar vector store missing ho toh direct LLM chalega)
        if retriever:
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | custom_prompt_template
                | llm_model
                | StrOutputParser()
            )
        else:
            # Fallback chain agar PDF database load nahi ho paaya
            rag_chain = (
                {"context": lambda x: "No context available.", "question": RunnablePassthrough()}
                | custom_prompt_template
                | llm_model
                | StrOutputParser()
            )

        # 4. Chain Execution
        try:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt)
            
            # Assistant Chat Render aur State Save
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Execution Engine Error: {str(e)}")


if __name__ == "__main__":
    main()