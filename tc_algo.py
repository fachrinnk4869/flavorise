import unittest
from sentence_transformers import SentenceTransformer
from AlgorithmClass import AlgorithmClass
from rag import embed_rag
import json


class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        json_file_path = './data/raw/cookpad_recipe_ayam.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.recipes = json.load(f)
        self.algorithm = AlgorithmClass()

    def test_output_mapping(self):
        recipes = self.recipes[:10]
        embeddings, embeding_ingredients = self.algorithm.generate_recipe_embeddings(
            recipes)
        mapping_result = self.algorithm.mapping_output(
            recipes, embeddings, embeding_ingredients)
        self.assertEqual(len(mapping_result), len(recipes))
        for mo in mapping_result:
            self.assertIsNotNone(mo.title)
            self.assertIsNotNone(mo.ingredients_vector)
            self.assertIsNotNone(mo.all_vector)
            self.assertIsNotNone(mo.final_vector)
            # print(mo)

    def test_algorithm(self):
        ''' Yang wajib declare di sini adalah:
            - self.algorithm = AlgorithmClass()
            - self.recipes = ... (list of recipes)
            - self.algorithm.mapping_output(...) -> self.algorithm.candidates
            - self.algorithm.mapping_input(...) -> self.algorithm.user_pref'''
        recipes = self.recipes[:50]
        embeddings_all, embeding_ingredients = self.algorithm.generate_recipe_embeddings(
            recipes)
        mapping_result = self.algorithm.mapping_output(
            recipes, embeddings_all, embeding_ingredients)
        self.assertEqual(len(mapping_result), len(recipes))

        input_text = "kecap"
        embedding_input = self.algorithm.generate_input_embedding(
            input_text)
        text_input, user_pref = self.algorithm.mapping_input(
            input_text, embedding_input)
        self.assertIsNotNone(user_pref)

        # simulasi rating user
        # generate rekomendasi pertama
        result = self.algorithm.first_generate_recipe()
        print(f"Rekomendasi awal: {result['title']}")
        print("Ingredients:")
        for ing in result['ingredients']:
            print(f"- {ing}")
        self.assertIsNotNone(result)
        rating = [-1, 1, 5, 3, -5]
        for step in range(5):
            result = self.algorithm.rating_recipe(rating=rating[step])
            self.assertIsNotNone(result)
            print(f"Rekomendasi step {step+1}: {result['title']}")
            print("Ingredients:")
            for ing in result['ingredients']:
                print(f"- {ing}")


if __name__ == '__main__':
    unittest.main()
