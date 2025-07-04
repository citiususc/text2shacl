# For HTML splitting
from unstructured.partition.html import partition_html

# For summarization
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from prompts import load_prompt_from_json

# For embedding
import uuid
from langchain_chroma import Chroma
from langchain_community.storage.redis import RedisStore

from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

from cloudflare import *

from ollama_functions import is_model_downloaded

from bs4 import BeautifulSoup

import os
import re
import pickle
import json


def _extract_base64_images(html_path):
    """
    Extract base64-encoded images from an HTML file.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    base64_images = []

    for img in soup.find_all("img"):
        src = img.get("src", "")
        if src.startswith("data:image/") and "base64," in src:
            match = re.search(r"base64,(.*)", src)
            if match:
                base64_images.append(match.group(1))

    return base64_images


def _split_html(file_path: str):
    """
    Split HTML file into text chunks, tables, and images.
    """
    chunks = partition_html(
        filename=file_path,
        skip_headers_and_footers=True
    )

    tables = []
    texts = []
    images = []

    # Separate tables and text
    for chunk in chunks:
        if "Table" in chunk.category:
            tables.append(chunk.metadata.text_as_html)
        else:
            texts.append(chunk.text)

    # Extract base64 images and upload them to a CDN
    for image in _extract_base64_images(file_path):
        url = upload_base64_image(image)
        images.append(url)

    return texts, tables, images


def _summarize(texts, tables, images):
    """
    Summarize text, tables, and images using language models.
    """

    # Summarize text
    prompt_text = load_prompt_from_json("prompts/rag_prompts.json", "text_summarization")
    batch_inputs = [{"content": text} for text in texts]
    chain = prompt_text | model | StrOutputParser()
    text_summaries = chain.batch(batch_inputs, {"max_concurrency": 3})

    # Summarize tables
    prompt_table = load_prompt_from_json("prompts/rag_prompts.json", "table_summarization")
    batch_inputs = [{"content": table} for table in tables]
    chain = prompt_table | model | StrOutputParser()
    table_summaries = chain.batch(batch_inputs, {"max_concurrency": 3})

    # Summarize images
    prompt_image = load_prompt_from_json("prompts/rag_prompts.json", "image_summarization")
    batch_inputs = [{"content": image} for image in images]
    chain = prompt_image | model | StrOutputParser()
    image_summaries = chain.batch(batch_inputs, {"max_concurrency": 3})

    return text_summaries, table_summaries, image_summaries


def _embedding(texts, text_summaries, tables, table_summaries, images, image_summaries, chroma_index):
    """
    Generate and store embeddings in a Chroma vector store and Redis doc store.
    """

    # Choose the embedding function depending on the model
    if isinstance(model, ChatOpenAI):
        embedding_function = OpenAIEmbeddings()
    else:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    filename = os.path.basename(chroma_index)

    # Create persistent Chroma vector store
    vectorstore = Chroma(
        collection_name=filename,
        embedding_function=embedding_function,
        persist_directory=chroma_index
    )

    # Redis for storing full documents
    store = RedisStore(redis_url="redis://localhost:6379")
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Store summarized texts
    if len(text_summaries) > 0:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Store summarized tables
    if len(table_summaries) > 0:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    # Store summarized images
    if len(image_summaries) > 0:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, images)))

    return retriever


def _save_cache(file_path, texts, tables, images, text_summaries, table_summaries, image_summaries):
    """
    Save processing results to disk (pickle).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump((texts, tables, images, text_summaries, table_summaries, image_summaries), f)


def _load_cache(file_path):
    """
    Load processing results from disk (pickle).
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _clear_cache(file_path):
    """
    Delete the cache file if it exists.
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def load_retriever(file, model_name, model_temperature, force_process):
    """
    Main entry point: processes or loads a document retriever from HTML input.
    """

    global account_id
    global bucket_name
    global model

    # Standardize filename for storage
    filename = os.path.splitext(os.path.basename(file))[0]
    filename = re.sub(r'[^a-z0-9-]', '-', filename.lower())

    # Load or create a language model
    if model_name == "gpt":
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=model_temperature
        )
    else:
        if is_model_downloaded("llama3:8b"):
            model = ChatOllama(
                model="llama3:8b",
                temperature=model_temperature
            )
        else:
            raise ValueError("The model 'llama3:8b' is not downloaded in Ollama.")

    # Initialize Cloudflare bucket credentials
    start_cloudflare()
    account_id, bucket_name = get_cloudflare_data()

    # Prepare cache and chroma vector store paths
    cache_file = f"processing_cache/{filename}_{model_name}_{model_temperature:.2f}.pkl"
    chroma_index = f"./chroma_db/{filename}_{model_name}_{model_temperature:.2f}"

    # Check cache
    if not force_process and os.path.exists(cache_file) and os.path.exists(chroma_index):
        print("📦 Loading data from cache and persistent vectorstore...")
        texts, tables, images, text_summaries, table_summaries, image_summaries = _load_cache(cache_file)

        if isinstance(model, ChatOpenAI):
            embedding_function = OpenAIEmbeddings()
        else:
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        vectorstore = Chroma(
            collection_name=f"{filename}_{model_name}_{model_temperature:.2f}",
            embedding_function=embedding_function,
            persist_directory=chroma_index
        )

        store = RedisStore(redis_url="redis://localhost:6379")

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key="doc_id",
        )
    else:
        print("🔄 Processing HTML...")
        _clear_cache(cache_file)
        texts, tables, images = _split_html(file)
        text_summaries, table_summaries, image_summaries = _summarize(texts, tables, images)
        _save_cache(cache_file, texts, tables, images, text_summaries, table_summaries, image_summaries)
        retriever = _embedding(texts, text_summaries, tables, table_summaries, images, image_summaries, chroma_index)
        print("✅ Data processed and stored.")

    return retriever
