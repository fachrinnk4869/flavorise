from typing import List
import numpy as np
from pipeline.get_embedding import get_dense_embeddings
# from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Helper functions
# --------------------------
from mapping import MappingOutput
import json


class AlgorithmClass:

    def __init__(self):

        # import recipes from json file
        self.top_k = 10
        self.selected = []
        self.user_pref = None
        self.candidates = []
        self.current_recipe = None

    def reset(self):
        self.selected = []
        self.user_pref = None
        self.candidates = []

    def generate_recipe_embeddings(self, recipes: List[dict]):
        ''' Digunakan untuk generate embedding dari list of recipes'''
        embeddings_all = []
        embeding_ingredients = []
        for recipe in recipes:
            if(('vector_all' in recipe) and ('values' in recipe)):
                embeddings_all.append(recipe['vector_all'] or [])
                embeding_ingredients.append(recipe['values'] or [])
        return embeddings_all, embeding_ingredients

    def generate_input_embedding(self, text_input: str):
        embedding_input = get_dense_embeddings(text_input)
        return embedding_input

    def mapping_input(self, text_input, embedding_input=None):
        if embedding_input is None:
            embedding_input = self.generate_input_embedding(text_input)
        self.user_pref = embedding_input
        return text_input, embedding_input

    def mapping_output(self, recipes, embeddings=None, embeding_ingredients=None) -> List[MappingOutput]:
        if embeddings is None or embeding_ingredients is None:
            embeddings, embeding_ingredients = self.generate_recipe_embeddings(
                recipes)

        mapping_result = [
            MappingOutput(title=recipe['title'],
                          image=recipe.get('image', None),
                          ingredients=recipe.get('ingredients', None),
                          steps=recipe.get('steps', None),
                          ingredients_vector=embeding_ingredients[i],
                          all_vector=embeddings[i],
                          final_vector=self.rerank_ingredients(
                              embeddings[i], embeding_ingredients[i], lambd=0.9)
                          )
            for i, recipe in enumerate(recipes)
        ]
        self.candidates = mapping_result.copy()
        return mapping_result

    def get_recipe(self):
        return self.current_recipe

    def first_generate_recipe(self):
        return self.rating_recipe(rating=0)

    def rating_recipe(self, rating):
        """Update preferensi user berdasarkan rating (0-1)."""
        # print("selected:", selected)
        try:
            reranked = self.mmr_rerank(lambd=0.7, top_k=1)[0]
        except IndexError:
            self.candidates = self.selected.copy()
            self.selected = []
            reranked = self.mmr_rerank(lambd=0.7, top_k=1)[0]
        # hanya rating top-1
        # print(f"\nIterasi {step+1}")
        # print(f"{reranked.title}. {step}")
        # print(f"Berikan rating untuk {top1_name} (-5->5): {rating[step]} ")
        self.user_pref = self.update_user_pref(
            self.user_pref, reranked.final_vector, rating, lr=0.8)

        self.current_recipe = {
            "steps": reranked.steps,
            "ingredients": reranked.ingredients,
            "image": reranked.image,
            "title": reranked.title
        }

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def update_user_pref(self, user_pref, item_embedding, rating, lr=0.5):
        """Update preferensi user berdasarkan rating (-5 - 5)."""
        # print(item_embedding)
        # print(user_pref)
        return user_pref + lr * rating * (item_embedding - user_pref)

    def mmr_rerank(self, lambd=0.7, top_k=1):
        """Maximal Marginal Relevance Reranking."""
        bests = []
        while len(bests) < top_k and self.candidates:
            scores = []
            for candidate in self.candidates:
                sim_to_user = self.cosine_similarity(
                    self.user_pref, candidate.final_vector)
                sim_to_selected = max([self.cosine_similarity(
                    candidate.final_vector, s.final_vector) for s in self.selected], default=0)
                score = lambd * sim_to_user - (1 - lambd) * sim_to_selected
                scores.append((score, candidate))
            scores.sort(key=lambda x: x[0], reverse=True)
            best = scores[0][1]
            self.selected.append(best)
            bests.append(best)
            self.candidates.remove(best)
        return bests

    def rerank_ingredients(self, embed_all, embed_ingredients, lambd=0.7):
        if not (0.0 <= lambd <= 1.0):
            raise ValueError("lambd must be in [0, 1]")

        e1 = np.asarray(embed_ingredients, dtype=np.float32)
        e2 = np.asarray(embed_all, dtype=np.float32)
        if e1.shape != e2.shape:
            raise ValueError(f"Shape mismatch: {e1.shape} vs {e2.shape}")

        out = lambd * e1 + (1.0 - lambd) * e2
        return out

    def matching_algorithm(self, list_rag) -> List[str]:
        # Dummy implementation of the matching algorithm
        # Replace this with the actual logic
        return [rag for rag in list_rag if "match" in rag]
