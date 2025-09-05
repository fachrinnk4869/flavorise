import json
from AlgorithmClass import AlgorithmClass
from pipeline.rag_pipeline import RAG_pipeline


class Datahandle:
    def get_recipes(self, query: str):
        # get top 50 recipe based on query from vector db
        output_recipes = RAG_pipeline(query)
        return output_recipes

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
