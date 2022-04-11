import tensorflow as tf
import numpy as np
from .utils import vocab as vc
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'model/bestbuy_model')

class Model:

    def __init__(self,path = filename,vocab = vc):
        self.model = tf.keras.models.load_model(filename,compile = False)
        self.vocab = vocab
        
    def get_labels(self,probilities):
    
        predicted_proba = [proba for proba in probilities]
        iterator = zip(predicted_proba,self.vocab)
        iterator_sorted = sorted(iterator,key = lambda tupl:tupl[0],reverse=True) ## sorting based on probabilities
        predicted_lables = [x for _,x in iterator_sorted]
        
        return predicted_lables

    def predict(self,text):
        text = tf.expand_dims(text,axis=0)
        prob = self.model.predict(text)
        return self.get_labels(prob[0])[:3]
