from typing import List
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Helper functions
# --------------------------


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def update_user_pref(user_pref, item_embedding, rating, lr=0.5):
    """Update preferensi user berdasarkan rating (0-1)."""
    # print(item_embedding)
    # print(user_pref)
    return user_pref + lr * rating * (item_embedding - user_pref)


def mmr_rerank(user_pref, candidates, lambd=0.7, top_k=1, selected=[]):
    """Maximal Marginal Relevance Reranking."""
    bests = []
    while len(bests) < top_k and candidates:
        scores = []
        for name, emb in candidates:
            sim_to_user = cosine_similarity(user_pref, emb)
            sim_to_selected = max([cosine_similarity(emb, s[1])
                                  for s in selected], default=0)
            score = lambd * sim_to_user - (1 - lambd) * sim_to_selected
            scores.append((score, (name, emb)))
        scores.sort(key=lambda x: x[0], reverse=True)
        best = scores[0][1]
        selected.append(best)
        bests.append(best)
        candidates.remove(best)
    return bests


def rerank_ingredients(embed_all, embed_ingredients, lambd=0.7):
    return embed_ingredients * lambd + embed_all * (1 - lambd)


def matching_algorithm(list_rag) -> List[str]:
    # Dummy implementation of the matching algorithm
    # Replace this with the actual logic
    return [rag for rag in list_rag if "match" in rag]
