import chromadb
import os
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

def get_vector_collection(workspace: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found")
    
    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-ada-002"
    )
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