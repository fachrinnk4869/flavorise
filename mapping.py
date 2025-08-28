from typing import List


class MappingOutput:

    def __init__(self, id: List, ingredients_vector: List, all_vector: List):
        self.id = id
        self.ingredients_vector = ingredients_vector
        self.all_vector = all_vector

    def to_matching_output(self):
        vector_all = {self.id[i]: embedding for i,
                      embedding in enumerate(self.all_vector)}
        vector_ingredients = {self.id[i]: embedding for i,
                              embedding in enumerate(self.ingredients_vector)}
        return vector_all, vector_ingredients
