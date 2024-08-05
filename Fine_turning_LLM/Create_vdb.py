from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def create_db_from_files(pdf_data_path):
    pypdf_loader_kwargs={'extract_images': True}
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader, loader_kwargs=pypdf_loader_kwargs)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = text_splitter.split_documents(documents)

    # qdrant = Qdrant.from_documents(
    # docs,
    # embedding_model,
    # path="./data",
    # collection_name="document_embeddings",
    # )

    # return qdrant

    db = FAISS.from_documents(documents=docs, embedding=embedding_model)
    db.save_local("data/db_faiss")
    return db


def get_similar_docs(query, qdrant):
    similar_docs = qdrant.similarity_search_with_score(query)

    return similar_docs

create_db_from_files("data")

# if __name__ == "__main__":
#     qd = create_db_from_files("data")
#     query = "how many bug BugsInPy currently have"
#     similar_docs = get_similar_docs(query, qd)
#     for doc, score in similar_docs:
#         print(f"text: {doc.page_content[:512]}\n")
#         print(f"score: {score}")
#         print("-" * 80)
#         print()