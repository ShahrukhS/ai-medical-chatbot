import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

CUSTOM_PROMPT_TEMPLATE = """
Use only the information provided in the context to answer the question.
If the answer cannot be found in the context, respond with: "I don't know."
Do not invent or infer information beyond the context.
Start the response concise and precise to the query.

Context:
{context}

Question:
{question}
"""
custom_rag_prompt = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.title("MediBot")
    st.write("Welcome to the Medical ChatBot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter your prompt:")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error('Failed to load the vector store')
            else:
                retriever = vectorstore.as_retriever(search_kwargs={'k':3})
                rag_chain = (
                    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
                    | custom_rag_prompt
                    | llm
                    | StrOutputParser()
                )

                response = rag_chain.invoke(prompt)
                st.chat_message('assistant').markdown(response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main()