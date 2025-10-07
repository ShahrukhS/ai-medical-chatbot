import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly, no small talk please.
"""

custom_rag_prompt = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k':3})

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

query = input("Enter your query: ") # "Which tables can I use for SOC codes under transition provision for salary."
response = rag_chain.invoke(query)
print('Response: ', response)
# print('Source Documents: ', response['source_documents'])