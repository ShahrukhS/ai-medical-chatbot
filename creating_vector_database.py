import warnings
warnings.filterwarnings("ignore")

# this video: https://www.youtube.com/watch?v=OP0FYjF-37c
# and kaggle for chroma DB. I have installed libraries and plan to use chromaDB instead of FAISS
# https://medium.com/@nermeen.abdelaziz/build-your-first-python-rag-using-chromadb-openai-d711db1abf66
# https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339


from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"

def load_documents():
  print("Loading documents into PyPDF objects ...")
  document_loader = DirectoryLoader(DATA_PATH,
                                    glob='*.pdf',
                                    loader_cls=PyPDFLoader)
  # Load PDF documents and return them as a list of Document objects
  return document_loader.load()

documents = load_documents()
print("Length of  documents:", len(documents))
print(documents[0])

def split_text(documents):
  
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # Size of each chunk in characters
    chunk_overlap=50, # Overlap between consecutive chunks
  )

  # Split documents into smaller chunks using text splitter
  chunks = text_splitter.split_documents(documents)
  print(f"Splitting {len(documents)} documents into {len(chunks)} chunks...")

  return chunks # Return the list of split text chunks

text_chunks = split_text(documents)

def get_embedding_model():
  embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
  embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
  return embeddings

DF_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, get_embedding_model())
db.save_local(DF_FAISS_PATH)