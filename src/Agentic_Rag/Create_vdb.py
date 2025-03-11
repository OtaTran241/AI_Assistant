from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, JSONLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from functools import partial
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

embedding_model_bge = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device}
)

# embedding_model_halong = HuggingFaceEmbeddings(
#     model_name="hiieu/halong_embedding",
#     model_kwargs={"device": device}
# )

def create_db_from_files(data_folder, embedding_model, save_path):
    loaders = [
        DirectoryLoader(data_folder, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(data_folder, glob="*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(data_folder, glob="*.json", loader_cls=partial(JSONLoader, jq_schema=".")),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=51)
    docs = text_splitter.split_documents(documents)

    print(f"Original documents: {len(documents)}")
    print(f"Split documents: {len(docs)}")

    if not docs:
        print("Không có tài liệu nào sau khi chia nhỏ. Kiểm tra lại dữ liệu đầu vào!")
        return None

    db = FAISS.from_documents(docs, embedding_model)

    db.save_local(f"{data_folder}/{save_path}")
    print(f"FAISS database saved to {data_folder}/{save_path}")

    return db

faiss_bge_path = "Vector_db/bge"
# faiss_halong_path = "Vector_db/halong"

db_bge = create_db_from_files("datas/RAG_datas", embedding_model_bge, faiss_bge_path)
# db_halong = create_db_from_files("datas/RAG_datas", embedding_model_halong, faiss_halong_path)