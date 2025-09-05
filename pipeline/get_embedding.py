# pipeline/get_embedding.py
import os
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
import requests

load_dotenv()

SILICONFLOW_URL_EMBEDDING = os.getenv("SILICONFLOW_URL_EMBEDDING")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
EMBED_DIM = int(os.getenv("EMBED_DIM")) if os.getenv("EMBED_DIM") else 1024

if not SILICONFLOW_URL_EMBEDDING:
    raise RuntimeError("SILICONFLOW_URL_EMBEDDING is not set")
if not SILICONFLOW_API_KEY:
    raise RuntimeError("SILICONFLOW_API_KEY is not set")

headers = {
    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
    "Content-Type": "application/json",
}

def get_dense_embeddings(text: str, dim_size: int = EMBED_DIM) -> list[float]:
    dim = dim_size or EMBED_DIM or 1024
    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": text,
        "encoding_format": "float",
        "dimensions": dim,
    }

    try:
        response = requests.post(
            SILICONFLOW_URL_EMBEDDING,
            json=payload,
            headers=headers
        )
        response.raise_for_status()

        data = response.json()

        # Validasi struktur response
        if "data" not in data or not data["data"]:
            raise ValueError("Response JSON tidak memiliki field 'data' atau kosong.")

        if "embedding" not in data["data"][0]:
            raise ValueError("Field 'embedding' tidak ditemukan di dalam 'data[0]'.")

        return data["data"][0]["embedding"]

    except requests.exceptions.RequestException as e:
        print(f"Error HTTP: {e}")
    except ValueError as e:
        print(f"Error data: {e}")
    except Exception as e:
        print(f"Error tidak terduga: {e}")

    return None

def get_sparse_embeddings(text: str, bm25_model: BM25Encoder, query_type: str = "search"):
    if query_type == "upsert":
        return bm25_model.encode_documents(text)
    return bm25_model.encode_queries(text)