import chromadb
import os
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

def get_vector_collection(workspace: str):
    backend = os.getenv("LLM_BACKEND", "openai").lower()
    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        model_name = "text-embedding-ada-002"
        embedding_function = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name
        )
    elif backend == "ollama":
        ollama_url = os.getenv("OLLAMA_EMBEDD_URL", "http://localhost:11434/api/embeddings")
        model_name = os.getenv("OLLAMA_EMBEDD_MODEL", "nomic-embed-text:latest")
        embedding_function = OllamaEmbeddingFunction(
            url=ollama_url,
            model_name=model_name
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend}. Use 'openai' or 'ollama'.")

    # Use backend and model in collection name to avoid dimension conflicts
    collection_name = f"rag_app_{backend}_{model_name.replace(':', '_').replace('-', '_').replace('.', '_')}_{workspace.replace(' ', '_').lower()}"
    chroma_client = chromadb.PersistentClient(path="./vector_store_data")
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits, file_name, workspace):
    collection = get_vector_collection(workspace)
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        meta = dict(split.metadata) if hasattr(split, 'metadata') else {}
        meta['file_name'] = file_name
        documents.append(split.page_content)
        metadatas.append(meta)
        ids.append(f"{file_name}_{idx}")
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids) 