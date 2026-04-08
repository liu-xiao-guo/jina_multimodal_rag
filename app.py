import os
import torch
import streamlit as st
from PIL import Image
from transformers import AutoModel
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import openai
import base64
from io import BytesIO

# Load environment variables from .env if exists
load_dotenv()

# -------------------------
# Config
# -------------------------
INDEX_NAME = "multimodal-index"
IMAGE_FOLDER = "./images"  # local image folder
TEXT_FOLDER = "./texts"    # local text folder
ES_URL = os.getenv("ES_URL", "https://localhost:9200")
ES_API_KEY = os.getenv("ES_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("GEMINI_FLASH_API_KEY")

# -------------------------
# Elasticsearch
# -------------------------
@st.cache_resource
def get_es():
    return Elasticsearch(ES_URL, verify_certs=False, api_key=ES_API_KEY)

es = get_es() # Initialize Elasticsearch client

# -------------------------
# Model loading
# -------------------------
@st.cache_resource
def load_model():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    return model, device

model, device = load_model()

# -------------------------
# LLM Client loading (OpenRouter)
# -------------------------
@st.cache_resource
def load_llm_client():
    if not OPENROUTER_API_KEY:
        st.error("GEMINI_FLASH_API_KEY not found in environment variables.")
        return None
    
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    return client

llm_client = load_llm_client()
LLM_MODEL_NAME = "google/gemini-3-flash-preview"

# -------------------------
# Index setup
# -------------------------
def create_index():
    if es.indices.exists(index=INDEX_NAME):
        return

    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "path": {"type": "keyword"},
                "caption": {"type": "text"},
                "vector_field": {
                    "type": "dense_vector",
                    "dims": 2048,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)

create_index()

# -------------------------
# Embedding helpers
# -------------------------
def embed_image(pil_image):
    with torch.inference_mode():
        vec = model.encode_image(
            images=[pil_image],
            task="retrieval",
            return_numpy=True
        )
    return vec[0]

def embed_text(text):
    with torch.inference_mode():
        vec = model.encode_text(
            texts=[text],
            task="retrieval",
            prompt_name="query",
            return_numpy=True
        )
    return vec[0]

# -------------------------
# Batch ingestion for images
# -------------------------
def ingest_image_folder(folder):
    docs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        path = os.path.join(folder, fname)
        image = Image.open(path).convert("RGB")
        vec = embed_image(image)

        docs.append({
            "_index": INDEX_NAME,
            "_source": {
                "filename": fname,
                "path": path,
                "caption": fname.replace("_", " "),
                "vector_field": vec.tolist(),
            }
        })

    if docs:
        from elasticsearch.helpers import bulk
        bulk(es, docs)

# -------------------------
# Batch ingestion for text files
# -------------------------
def ingest_text_folder(folder):
    docs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        vec = embed_text(text)

        docs.append({
            "_index": INDEX_NAME,
            "_source": {
                "filename": fname,
                "path": path,
                "caption": text[:500],
                "vector_field": vec.tolist(),
            }
        })

    if docs:
        from elasticsearch.helpers import bulk
        bulk(es, docs)

# -------------------------
# KNN search only
# -------------------------
def knn_search(query, k=10):
    vec = embed_text(query)
    body = {
        "size": k,
        "query": {
            "knn": {
                "field": "vector_field",
                "query_vector": vec.tolist(),
                "k": k,
                "num_candidates": 50
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=body)
    return res["hits"]["hits"]

# -------------------------
# Image to Base64 helper
# -------------------------
def pil_to_base64(image, format="jpeg"):
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format};base64,{img_str}"

# -------------------------
# RAG Augmentation
# -------------------------
def generate_rag_response(user_query: str, k: int = 3):
    """
    Retrieves top K documents, creates a multimodal prompt, and generates a response from Gemini via OpenRouter.
    """
    st.write(f"Searching for top {k} relevant documents for RAG...")
    results = knn_search(user_query, k=k)
    
    if not results:
        st.warning("No relevant documents found for RAG.")
        return

    # Build the multimodal prompt for OpenAI-compatible API
    content_parts = []
    text_context = "Based on the following information:\n"

    for hit in results:
        src = hit["_source"]
        path = src.get("path", "")
        if path and os.path.exists(path) and path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            text_context += f"- Image: {src.get('filename', 'N/A')}\n"
            try:
                img = Image.open(path).convert("RGB")
                base64_image = pil_to_base64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": base64_image}
                })
            except Exception as e:
                st.error(f"Could not load image {path}: {e}")
        else:
            text_context += f"- Text Content: {src.get('caption', 'N/A')}\n"

    text_context += f"\nAnswer the question: {user_query}"
    content_parts.insert(0, {"type": "text", "text": text_context})

    messages = [{"role": "user", "content": content_parts}]

    st.subheader("Gemini Flash Multimodal Prompt:")
    st.json(messages)

    if llm_client:
        with st.spinner("Gemini Flash is generating a response via OpenRouter..."):
            try:
                response = llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=messages,
                    max_tokens=1024,
                )
                st.markdown("**LLM Generated Response:**")
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error generating response from OpenRouter: {e}")

# -------------------------
# Streamlit UI
# -------------------------
st.title("🖼️📄 Multimodal Image & Text KNN Search")

# Batch ingestion buttons
st.subheader("Ingest Data")
if st.button("📥 Ingest image folder"):
    ingest_image_folder(IMAGE_FOLDER)
    st.success("Images ingested successfully")

if st.button("📥 Ingest text folder"):
    ingest_text_folder(TEXT_FOLDER)
    st.success("Text files ingested successfully")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("⚠️ Delete & Re-ingest All"):
        with st.spinner("Deleting index and re-ingesting all data..."):
            if es.indices.exists(index=INDEX_NAME):
                es.indices.delete(index=INDEX_NAME)
                st.toast(f"Index '{INDEX_NAME}' deleted.")
            
            create_index()
            st.toast("Index created.")
            ingest_image_folder(IMAGE_FOLDER)
            st.toast("Images ingested.")
            ingest_text_folder(TEXT_FOLDER)
            st.toast("Texts ingested.")
        st.success("All data has been re-ingested successfully!")

# Search box for typing text queries
st.subheader("Search")
user_query = st.text_input("Type your search query here", key="search_query")
k_value = st.slider("Number of results to retrieve (K)", min_value=1, max_value=10, value=4)

if user_query:
    st.subheader(f"Retrieval Results (Top {k_value})")
    retrieval_results = knn_search(user_query, k=k_value)

    if retrieval_results:
        cols_per_row = 3
        for i in range(0, len(retrieval_results), cols_per_row):
            row = retrieval_results[i:i + cols_per_row]
            cols = st.columns(len(row), gap="medium")

            for col, hit in zip(cols, row):
                src = hit["_source"]
                path = src.get("path", "")

                if path and os.path.exists(path) and path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    col.image(path, caption=f"{src.get('filename', '')}", width=200)
                else:
                    col.write(f"**[TEXT]**\n\n{src.get('caption', '')}")

                col.write(f"Score: {hit['_score']:.3f}")
    else:
        st.info("No relevant documents found in the index.")

    st.subheader("RAG System Output")
    st.write("---")
    generate_rag_response(user_query, k=k_value)
