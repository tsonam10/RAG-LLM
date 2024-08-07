import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from langchain_community.llms import ctransformers
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import CTransformers 

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs ={'normalize_embeddings': False})

    return embeddings 

def load_documents():
    loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls =PyPDFLoader)
    document =loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    return texts 

def add_to_chroma(texts, embeddings):
    vector_store =Chroma.from_documents(load_documents(), embeddings,
                                        collection_metadata={"hnsw:space": "cosine"}, 
                                        persist_directory="stores/pet_cosine")

    print("Vector Store Created........")
    return vector_store 


def create_llm():
    model ="neural-chat-7b-v3-1.Q4_K_M.gguf"
    llm=CTransformers(
    model = model, 
    model_type ='llama',
    max_new_tokens=1024,
    temperature=0.1,
    top_p=0.95,
    top_k=50,
    repetition_penality=1.1)
    print("LLM Inititalized.......")
