import unittest
from sentence_transformers import SentenceTransformer
from rag import embed_rag
from algorithm import matching_algorithm, rerank_ingredients, mmr_rerank, update_user_pref
import json


class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 384  # for MiniLM-L12-v2
        # multilingual lm
        self.embeding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        # import recipes from json file
        self.top_k = 10
        json_file_path = './data/raw/cookpad_recipe_ayam.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.recipes = json.load(f)

    def test_init(self):
        sentences = ["This is an example sentence",
                     "Each sentence is converted"]
        embeddings = self.embeding_model.encode(sentences)
        self.assertEqual(embeddings.shape, (len(sentences), self.embed_dim))

    def test_import_rag(self):
        list_rag = embed_rag()
        self.assertGreater(len(list_rag), 0)

    def test_embed_rag(self):
        list_rag = embed_rag()
        embeddings = self.embeding_model.encode(list_rag)
        print("test embed rag", embeddings.shape)
        self.assertEqual(embeddings.shape, (len(list_rag), self.embed_dim))

    def test_embed_recipes(self):
        recipes = self.recipes[:10]
        recipe_texts = [
            recipe['title'] + ' ' +
            " ".join(recipe['ingredients']) + ' ' +
            " ".join(step['text'] for step in recipe["steps"])
            for recipe in recipes
        ]
        embeddings = self.embeding_model.encode(recipe_texts)
        print("test embed recipes", embeddings.shape)
        self.assertEqual(embeddings.shape, (len(recipes), self.embed_dim))

    def test_mapping_output(self):
        from mapping import MappingOutput
        import numpy as np
        recipes = self.recipes[:10]
        # create embedding for all metadata
        recipe_texts = [
            recipe['title'] + ' ' +
            " ".join(recipe['ingredients']) + ' ' +
            " ".join(step['text'] for step in recipe["steps"])
            for recipe in recipes
        ]
        embeddings = self.embeding_model.encode(recipe_texts)
        # crate embedding for ingredients only
        embeding_ingredients = self.embeding_model.encode(
            [" ".join(recipe['ingredients']) for recipe in recipes])
        id = [recipe['title'] for recipe in recipes]
        print(id)
        mo = MappingOutput(id=id,
                           ingredients_vector=embeding_ingredients,
                           all_vector=embeddings)
        vector_all, vector_ingredients = mo.to_matching_output()
        self.assertEqual(len(vector_all), len(recipes))
        self.assertEqual(len(vector_ingredients), len(recipes))
        self.assertIsInstance(vector_all, dict)
        self.assertIsInstance(vector_ingredients, dict)

    def test_matching_algorithm2(self):
        # --------------------------
        # Simulasi
        # --------------------------
        from mapping import MappingOutput
        recipes = self.recipes[:10]
        # create embedding for all metadata
        recipe_texts = [
            recipe['title'] + ' ' +
            " ".join(recipe['ingredients']) + ' ' +
            " ".join(step['text'] for step in recipe["steps"])
            for recipe in recipes
        ]
        embeddings = self.embeding_model.encode(recipe_texts)
        # crate embedding for ingredients only
        embeding_ingredients = self.embeding_model.encode(
            [" ".join(recipe['ingredients']) for recipe in recipes])
        id = [recipe['title'] for recipe in recipes]
        print(id)
        mo = MappingOutput(id=id,
                           ingredients_vector=embeding_ingredients,
                           all_vector=embeddings)
        imgs_all, imgs_ingredients = mo.to_matching_output()
        # fungsi rerank dengan mmr
        user_pref = embeddings[0]   # preferensi awal user
        imgs = {name: rerank_ingredients(
            imgs_all[name], imgs_ingredients[name], lambd=0.7) for name in imgs_all}
        candidates = list(imgs.items())

        selected = []
        rating = [-1, 1, 5, 3, -5]  # simulasi rating user
        for step in range(5):  # iterasi 5x
            # print("selected:", selected)
            reranked = mmr_rerank(user_pref, candidates,
                                  lambd=0.7, top_k=1, selected=selected)

            # hanya rating top-1
            top1_name, top1_emb = reranked[0]
            print(f"\nIterasi {step+1}")
            for rank, (name, _) in enumerate(reranked, 1):
                print(f"{rank}. {name}")
            print(f"Berikan rating untuk {top1_name} (-5->5): {rating[step]} ")
            user_pref = update_user_pref(
                user_pref, top1_emb, rating[step], lr=0.8)

    # def test_matching_algorithm(self):
    #     # --------------------------
    #     # Simulasi
    #     # --------------------------
    #     recipes = self.recipes[:10]
    #     # create embedding for all metadata
    #     recipe_texts = [
    #         recipe['title'] + ' ' +
    #         " ".join(recipe['ingredients']) + ' ' +
    #         " ".join(step['text'] for step in recipe["steps"])
    #         for recipe in recipes
    #     ]
    #     embeddings = self.embeding_model.encode(recipe_texts)
    #     # crate embedding for ingredients only
    #     embeding_ingredients = self.embeding_model.encode(
    #         [" ".join(recipe['ingredients']) for recipe in recipes])

    #     # fungsi rerank dengan mmr
    #     user_pref = embeddings[0]   # preferensi awal user
    #     imgs_all = {f"ayam_{i}": embedding for i,
    #                 embedding in enumerate(embeddings[1:])}
    #     imgs_ingredients = {f"ayam_{i}": embedding for i,
    #                         embedding in enumerate(embeding_ingredients[1:])}
    #     imgs = {name: rerank_ingredients(
    #         imgs_all[name], imgs_ingredients[name], lambd=0.7) for name in imgs_all}
    #     candidates = list(imgs.items())

    #     selected = []
    #     rating = [-1, 1, 5, 3, -5]  # simulasi rating user
    #     for step in range(5):  # iterasi 5x
    #         # print("selected:", selected)
    #         reranked = mmr_rerank(user_pref, candidates,
    #                               lambd=0.7, top_k=1, selected=selected)

    #         # hanya rating top-1
    #         top1_name, top1_emb = reranked[0]
    #         print(f"\nIterasi {step+1}")
    #         for rank, (name, _) in enumerate(reranked, 1):
    #             print(f"{rank}. {name}")
    #         print(f"Berikan rating untuk {top1_name} (-5->5): {rating[step]} ")
    #         user_pref = update_user_pref(
    #             user_pref, top1_emb, rating[step], lr=0.8)


if __name__ == '__main__':
    unittest.main()
