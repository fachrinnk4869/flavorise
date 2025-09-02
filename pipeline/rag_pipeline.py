# pipeline/rag_pipeline.py
import os
from collections import defaultdict

import asyncio
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from pinecone.grpc import PineconeGRPC as Pinecone
from pipeline.get_embedding import get_dense_embeddings, get_sparse_embeddings
from pipeline.bm25_model import load_bm25_model

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# load env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
NAME_PINECONE_DENSE = os.getenv('NAME_PINECONE_DENSE')
NAME_PINECONE_SPARSE = os.getenv('NAME_PINECONE_SPARSE')
SILICONFLOW_URL_RERANK = os.getenv('SILICONFLOW_URL_RERANK')
SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY')

NAMESPACE = os.getenv('NAMESPACE')
TOP_K = 10
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else 1024

# config
pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(name=NAME_PINECONE_DENSE)
index_sparse = pc.Index(name=NAME_PINECONE_SPARSE)
bm25 = load_bm25_model()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
async def _pinecone_query_dense(vector):
    return await asyncio.to_thread(
        index_dense.query,
        namespace=NAMESPACE,
        vector=vector,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False,
    )
    

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
async def _pinecone_query_sparse(sparse_vector):
    return await asyncio.to_thread(
        index_sparse.query,
        namespace=NAMESPACE,
        sparse_vector=sparse_vector,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )

async def search_dense_index_async(text: str):
    vec = await get_dense_embeddings(text, EMBED_DIM)
    dense_response = await _pinecone_query_dense(vec)
    matches = dense_response.get("matches", []) or []
    return [{
        "id": item.get("id"),
        "similarity": item.get('score', 0.0),
        "category": item['metadata'].get("category", ''),
    } for item in matches]

async def search_sparse_index_async(text: str):
    sp = get_sparse_embeddings(text=text, bm25_model=bm25, query_type='search')
    sparse_response = await _pinecone_query_sparse(sp)
    matches = sparse_response.get("matches", []) or []
    return [{
        "id": item.get("id"),
        "similarity": item.get('score', 0.0),
        "category": item['metadata'].get("category", ''),
    } for item in matches]

"""
RRF score(d) = Σ 1/(k+rank(d)) where k is between 1-60 where d is document
"""
def rrf_fusion(dense_results, sparse_results, k=60, top_n=TOP_K):
    scores = defaultdict(float)

    # add rrf score from dense result
    for rank, res in enumerate(dense_results, 1):
        doc_id = res['id']
        scores[doc_id] += 1/(k + rank)

    # add rrf score from dense result
    for rank, res in enumerate(sparse_results, 1):
        doc_id = res['id']
        scores[doc_id] += 1/(k + rank)

    # sort by rrf score desc
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    all_results = {r['id']: r for r in dense_results + sparse_results}
    fused_results = [all_results[doc_id] for doc_id, _ in fused[:top_n]]

    return fused_results

async def RAG_pipeline_async(query: str):
    # create task to search data from pinecone simultaneously
    dense_results, sparse_results = await asyncio.gather(
        search_dense_index_async(query),
        search_sparse_index_async(query),
        return_exceptions=False,
    )

    # fusion filter and non filter result
    fused_results = rrf_fusion(dense_results, sparse_results)

    return fused_results