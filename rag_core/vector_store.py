import chromadb
from langchain_openai import OpenAIEmbeddings

def get_vector_collection(workspace: str):
    embedding_function = OpenAIEmbeddings()
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    collection_name = f"rag_app_{workspace.replace(' ', '_').lower()}"
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits, file_name, workspace):
    collection = get_vector_collection(workspace)
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids) 