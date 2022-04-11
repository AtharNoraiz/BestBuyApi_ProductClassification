from .classification import Model
class category:
    def __init__(self,model):

        self.model= model()

    def get_categories(self,data):

        text = data['name']+data['description']
        categories_list = self.model.predict(text)
        return {"categories":categories_list}



products = category(Model)
