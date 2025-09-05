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

    def reset(self):
        self.selected = []
        self.user_pref = None
        self.candidates = []

    def generate_recipe_embeddings(self, recipes: List[dict]):
        ''' Digunakan untuk generate embedding dari list of recipes'''
        # create embedding for all metadata
        embeding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        recipe_texts = [
            recipe['title'] + ' ' +
            " ".join(recipe['ingredients']) + ' ' +
            " ".join(step['text'] for step in recipe["steps"])
            for recipe in recipes
        ]
        embeddings = embeding_model.encode(recipe_texts)
        # crate embedding for ingredients only
        embeding_ingredients = embeding_model.encode(
            [" ".join(recipe['ingredients']) for recipe in recipes])
        return embeddings, embeding_ingredients

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

    def first_generate_recipe(self):
        return self.rating_recipe(rating=0)

    def rating_recipe(self, rating):
        """Update preferensi user berdasarkan rating (0-1)."""
        # print("selected:", selected)
        reranked = self.mmr_rerank(lambd=0.7, top_k=1)[0]
        # hanya rating top-1
        # print(f"\nIterasi {step+1}")
        # print(f"{reranked.title}. {step}")
        # print(f"Berikan rating untuk {top1_name} (-5->5): {rating[step]} ")
        self.user_pref = self.update_user_pref(
            self.user_pref, reranked.final_vector, rating, lr=0.8)
        return {
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
        return embed_ingredients * lambd + embed_all * (1 - lambd)

    def matching_algorithm(self, list_rag) -> List[str]:
        # Dummy implementation of the matching algorithm
        # Replace this with the actual logic
        return [rag for rag in list_rag if "match" in rag]
