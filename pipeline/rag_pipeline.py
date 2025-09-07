# pipeline/rag_pipeline.py
import os
from collections import defaultdict
from dotenv import load_dotenv
import json
from functools import lru_cache
from pinecone.grpc import PineconeGRPC as Pinecone
from pipeline.get_embedding import get_dense_embeddings, get_sparse_embeddings
from pipeline.bm25_model import load_bm25_model
import numpy as np

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
NAMESPACE2 = os.getenv('NAMESPACE2')
TOP_K = 50
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else 1024
RECIPES_FOLDER = 'data/clean'
SIMILARITY_THRESHOLD = 0.7

# config
pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(name=NAME_PINECONE_DENSE)
index_sparse = pc.Index(name=NAME_PINECONE_SPARSE)
bm25 = load_bm25_model()

def search_dense_index(text: str):
    vec = get_dense_embeddings(text, EMBED_DIM)
    dense_response = index_dense.query(
        namespace=NAMESPACE,
        vector= vec,
        top_k=TOP_K,
        include_metadata=True,
        include_values=True
    )
    matches = dense_response.get("matches", []) or []
    results = [{
        "id": item.get("id"),
        "similarity": item.get('score', 0.0),
        "category": (item.get('metadata') or {}).get("category", ''),
        "values": item.get('values')
    } for item in matches]

    # filter threshold
    if any((r.get("similarity") or 0.0) > SIMILARITY_THRESHOLD for r in results):
        results = [r for r in results if (r.get("similarity") or 0.0) > SIMILARITY_THRESHOLD]

    return results

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _fetch_dense_values_by_ids(ids, query_vec=None):
    """
    Ambil dense embedding values dari index_dense untuk sekumpulan ID.
    Return: dict[id] -> values (list of floats)
    """
    if not ids:
        return {}, {}

    dense_fetch = index_dense.fetch(ids=ids, namespace=NAMESPACE) or {}

    vectors_obj = dense_fetch.vectors or {}

    if isinstance(vectors_obj, dict):
        vectors_map = vectors_obj
    elif isinstance(vectors_obj, list):
        vectors_map = {v.get("id"): v for v in vectors_obj if v and v.get("id")}
    else:
        vectors_map = {}

    # Ekstrak values
    out = {}
    sim = {}
    for _id, rec in vectors_map.items():
        if not rec:
            continue
        values = rec.get("values")
        if values is None:
            values = (rec.get("vector") or {}).get("values")
        if values is not None:
            out[_id] = values
            sim[_id] = cosine_similarity(values, query_vec)
    
    return out, sim

def search_sparse_index(text: str):
    sp = get_sparse_embeddings(text=text, bm25_model=bm25, query_type='search')
    sparse_response = index_sparse.query(
        namespace=NAMESPACE,
        sparse_vector=sp,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    matches = sparse_response.get("matches", []) or []

    # get id from sparse search
    ids = [m.get("id") for m in matches if m.get("id")]

    # fetch dense embedding from id above
    query_dense_vec = get_dense_embeddings(text, EMBED_DIM)
    id_to_dense_values, id_to_sim = _fetch_dense_values_by_ids(ids, query_dense_vec)

    # map output
    results = []
    for item in matches:
        _id = item.get("id")
        results.append({
            "id": _id,
            "similarity": id_to_sim.get(_id, 0.0),
            "category": (item.get("metadata") or {}).get("category", ''),
            "values": id_to_dense_values.get(_id)
        })

    # filter threshold
    if any((r.get("similarity") or 0.0) > SIMILARITY_THRESHOLD for r in results):
        results = [r for r in results if (r.get("similarity") or 0.0) > SIMILARITY_THRESHOLD]

    return results

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

def batch_fetch_all_vectors(ids):
    """Fetch the ALL-TEXT vectors for a list of IDs from NAMESPACE2 in one RPC.
    Returns a dict: id -> {id, values, metadata}
    """
    if not ids:
        return {}
    if not NAMESPACE2:
        return {}

    fetched = index_dense.fetch(ids=ids, namespace=NAMESPACE2)
    return fetched.vectors

@lru_cache(maxsize=1)
def _build_recipe_lookup(folder_path: str):
    lookup = {}
    if not folder_path or not os.path.isdir(folder_path):
        return lookup

    try:
        with os.scandir(folder_path) as it:
            for entry in it:
                if not entry.is_file() or not entry.name.endswith('.json'):
                    continue
                try:
                    with open(entry.path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue

                items = data if isinstance(data, list) else [data]
                for obj in items:
                    if not isinstance(obj, dict):
                        continue
                    rid = obj.get('id')
                    if not rid:
                        continue
                    lookup[rid] = {
                        'url': obj.get('url'),
                        'title': obj.get('title'),
                        'image': obj.get('image'),
                        'ingredients': obj.get('ingredients'),
                        'steps': obj.get('steps'),
                    }
    except Exception:
        return lookup

    return lookup

def RAG_pipeline(query: str):
    # 1. Try sparse search first
    sparse_results = search_sparse_index(query)
    results = sparse_results
    ids = [r['id'] for r in sparse_results]

    # 2. Fallback: if no sparse matches, try dense search
    if not ids:
        dense_results = search_dense_index(query)
        results = dense_results
        ids = [r['id'] for r in dense_results]

    # 3. If still no results, return []
    if not ids:
        return []

    # 4. Fetch "all-text" vectors
    fetched_all = batch_fetch_all_vectors(ids)

    # 5. Load recipe metadata
    recipe_lookup = _build_recipe_lookup(RECIPES_FOLDER)

    # 6. Merge vectors + recipe metadata into results
    all_data = []
    for r in results:
        _id = r['id']
        all_payload = fetched_all.get(_id) if fetched_all else None
        r['vector_all'] = all_payload.get('values') if all_payload else None

        recipe = recipe_lookup.get(_id)
        if recipe:
            r.update({
                'url': recipe.get('url'),
                'title': recipe.get('title'),
                'image': recipe.get('image'),
                'ingredients': recipe.get('ingredients'),
                'steps': recipe.get('steps'),
            })
        else:
            r.update({
                'url': None,
                'title': None,
                'image': None,
                'ingredients': None,
                'steps': None,
            })

        all_data.append(r)

    return all_data
