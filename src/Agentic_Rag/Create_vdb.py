from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, JSONLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import faiss

# ✅ Sử dụng GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# embedding_model = HuggingFaceEmbeddings(
#     model_name="vinai/phobert-large",
#     model_kwargs={"device": device}
# )

embedding_model = HuggingFaceEmbeddings(
    model_name="dangvantuan/vietnamese-embedding",
    model_kwargs={"device": device}
)

def create_db_from_files(data_folder):
    pypdf_loader_kwargs = {'extract_images': True}
    loaders = [
        DirectoryLoader(data_folder, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader, loader_kwargs=pypdf_loader_kwargs),
        DirectoryLoader(data_folder, glob="*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(data_folder, glob="*.json", loader_cls=JSONLoader)
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(documents=docs, embedding=embedding_model)

    if device == "cuda":
        print("Chuyển FAISS index sang GPU...")
        res = faiss.StandardGpuResources()
        db.index = faiss.index_cpu_to_gpu(res, 0, db.index)

    db.save_local(f"{data_folder}/Vector_db")
    return db

create_db_from_files("datas/Rag_datas")