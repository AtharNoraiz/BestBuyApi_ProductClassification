import os
import tensorflow as tf
import json
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}','-','-'])

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'vocab_data.json')
with open(filename, 'r') as fp:
    data = json.load(fp)
vocab=data['vocab']
@tf.keras.utils.register_keras_serializable()
def clean_text(input_data):
    
    lowercase = tf.strings.lower(input_data)
    cleaned_stopwords=tf.strings.regex_replace(lowercase, r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*',"")
    cleaned_text=tf.strings.regex_replace(cleaned_stopwords,'[%s]' % re.escape(string.punctuation),'')
    return cleaned_text