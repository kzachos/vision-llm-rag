import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Patch sys.modules before anything else
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import csv
import tempfile
from io import StringIO
import yaml

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import refactored modules
from rag_core.vector_store import get_vector_collection, add_to_vector_collection
from rag_core.document import process_document
from rag_core.rerank import re_rank_cross_encoders
from rag_core.llm import call_llm

# Load workspaces from config file
def load_workspaces():
    with open("workspaces.yaml") as f:
        config = yaml.safe_load(f)
    return config.get("workspaces", ["Default Workspace"])

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def query_semantic_cache(query: str, n_results: int = 1, threshold: float = 80.0):
    vector_store = get_vector_collection()
    results = vector_store.similarity_search_with_score(query, k=n_results)
    if not results:
        return None
    match_percentage = (1 - abs(results[0][1])) * 100
    if match_percentage >= threshold:
        return results
    return None


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def get_document_info():
    try:
        collection = get_vector_collection()
        all_results = collection.get()
        if not all_results['ids']:
            return 0, []
        document_names = set()
        for doc_id in all_results['ids']:
            doc_name = doc_id.rsplit('_', 1)[0]
            document_names.add(doc_name)
        total_documents = len(all_results['ids'])
        return total_documents, sorted(list(document_names))
    except Exception as e:
        st.error(f"Error getting document info: {str(e)}")
        return 0, []


if __name__ == "__main__":
    workspaces = load_workspaces()
    with st.sidebar:
        st.set_page_config(page_title="VISION: Violence, Health & Society QnA Platform")
        st.markdown("""
        # 🛡️ VISION: Violence, Health & Society
        
        **A UKPRP-funded research consortium platform for secure, multi-tenant QnA over violence and health data.**
        
        Select your workspace below. Each workspace is isolated for its partner group:
        - Central Government
        - Local Government
        - Third Sector
        """)
        workspace = st.selectbox(
            "Select VISION workspace",
            workspaces,
            help="Choose which VISION workspace to upload/query files for."
        )
        st.header("\U0001F4D1 Upload Evidence or Cache")
        uploaded_files = st.file_uploader(
            "**Upload PDF (evidence) or CSV (Q&A cache) files**",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            help="Upload primary evidence (PDF) or Q&A cache (CSV) for your workspace.",
        )
        upload_option = st.radio(
            "Upload type:",
            options=["Primary Evidence", "Cache (Q&A pairs)"],
            help="Choose 'Primary Evidence' for new documents, or 'Cache' for Q&A pairs (CSV).",
        )
        if uploaded_files and upload_option == "Primary Evidence":
            csv_files = [f for f in uploaded_files if f.name.split(".")[-1].lower() == "csv"]
            if csv_files:
                st.error("CSV files are only allowed for 'Cache' upload type.")
                sys.exit(1)
        process = st.button(
            "\u26A1\uFE0F Ingest Evidence/Cache",
        )
        if uploaded_files and process:
            if upload_option == "Cache (Q&A pairs)":
                csv_files = [f for f in uploaded_files if f.name.split(".")[-1].lower() == "csv"]
                if csv_files:
                    data = csv_files[0].getvalue().decode("utf-8")
                    csv_reader = csv.DictReader(StringIO(data))
                    docs = []
                    for row in csv_reader:
                        docs.append(
                            Document(page_content=row["question"], metadata={"answer": row["answer"]})
                        )
                    vector_store = get_vector_collection(workspace)
                    vector_store.add_documents(docs)
                    st.success("Q&A cache added to VISION workspace!")
                else:
                    st.error("No CSV file found for cache upload.")
                    sys.exit(1)
            else:
                pdf_files = [f for f in uploaded_files if f.name.split(".")[-1].lower() == "pdf"]
                if not pdf_files:
                    st.error("No PDF files found for primary evidence upload.")
                    sys.exit(1)
                for pdf_file in pdf_files:
                    normalize_uploaded_file_name = pdf_file.name.translate(
                        str.maketrans({"-": "_", ".": "_", " ": "_"})
                    )
                    all_splits = process_document(pdf_file)
                    add_to_vector_collection(all_splits, normalize_uploaded_file_name, workspace)
        st.divider()
        st.header("\U0001F4CA Workspace Document Info")
        try:
            collection = get_vector_collection(workspace)
            all_results = collection.get()
            if not all_results['ids']:
                doc_names = []
            else:
                document_names = set()
                for doc_id in all_results['ids']:
                    doc_name = doc_id.rsplit('_', 1)[0]
                    document_names.add(doc_name)
                doc_names = sorted(list(document_names))
        except Exception as e:
            st.error(f"Error getting document info: {str(e)}")
            doc_names = []
        st.metric("Files in workspace", len(doc_names))
        if doc_names:
            st.subheader("\U0001F4C1 Uploaded Files:")
            for i, doc_name in enumerate(doc_names, 1):
                st.write(f"{i}. {doc_name}")
        else:
            st.info("No documents uploaded to this VISION workspace yet.")

    st.header("\U0001F5E3\uFE0F VISION QnA")
    st.markdown("""
    **Ask a question about your uploaded evidence or Q&A cache.**
    
    _This platform is for research and policy support within the VISION consortium. All answers are based solely on your uploaded documents._
    """)
    prompt = st.text_area("**Enter your question:**")
    ask = st.button(
        "\U0001F525 Get VISION Answer",
    )

    if ask and prompt:
        try:
            vector_store = get_vector_collection(workspace)
            results = vector_store.similarity_search_with_score(prompt, k=1)
            if results:
                match_percentage = (1 - abs(results[0][1])) * 100
                if match_percentage >= 80.0:
                    st.write(results[0][0].metadata["answer"].replace("\\n", "\n"))
                sys.exit(0)
        except Exception:
            pass
        collection = get_vector_collection(workspace)
        results = collection.query(query_texts=[prompt], n_results=10)
        context = results.get("documents")[0]
        if not context:
            st.write("No relevant evidence found in your VISION workspace.")
            sys.exit(1)
        relevant_text, relevant_text_ids, relevant_metadata = re_rank_cross_encoders(prompt, context, results.get("metadatas", []))
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)
        with st.expander("See retrieved documents"):
            st.write(results)
        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)
