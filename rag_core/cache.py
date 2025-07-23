# from langchain_ollama import OllamaEmbeddings
# from langchain_redis import RedisConfig, RedisVectorStore

# def get_redis_store(workspace: str):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
#     index_name = f"cached_contents_{workspace.replace(' ', '_').lower()}"
#     return RedisVectorStore(
#         embeddings,
#         config=RedisConfig(
#             index_name=index_name,
#             redis_url="redis://localhost:6379",
#             distance_metric="COSINE",
#             metadata_schema=[{"name": "answer", "type": "text"}],
#         ),
#     ) 