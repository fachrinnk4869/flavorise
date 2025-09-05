import json
from AlgorithmClass import AlgorithmClass


class Datahandle:
    def get_recipes(self):
        # ini ganti dengan recipes hasil rag
        json_file_path = './data/raw/cookpad_recipe_ayam.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        return recipes[:2]

    # ini ganti sama embedding sesuai maneh pake model apa untuk embeddingnya
    def get_embeddings_input(self, text_input):
        '''
        Input: text input (str)
        Output: embedding_input '''
        return AlgorithmClass().generate_input_embedding(
            text_input)

    # ini ganti sama embedding recipe dari maneh
    def get_embeddings_recipe(self, recipes):
        '''
        Input: list of recipes (dict)
        Output: embeddings_all, embeding_ingredients
        '''
        return AlgorithmClass().generate_recipe_embeddings(
            recipes)
