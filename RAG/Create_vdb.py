from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, JSONLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def create_db_from_files(data_folder):
    loaders = [
        DirectoryLoader(data_folder, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(data_folder, glob="*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(data_folder, glob="*.json", loader_cls=JSONLoader)
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(documents=docs, embedding=embedding_model)
    db.save_local(f"{data_folder}/Vector_db")
    return db

create_db_from_files("datas/Rag_datas")