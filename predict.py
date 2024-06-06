from sklearn.cluster import KMeans
import numpy as np
from joblib import load
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import pandas as pd

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    cleaned_data = data[data['speaker'] == 'Participant']
    return cleaned_data

def preprocess_text(texts, tokenizer, max_len=128):
    tokens = tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

def build_text_model():
    text_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    return text_model

def main(file_path, model_path):
    data = load_and_clean_data(file_path)
    texts = data['value']
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    input_ids, attention_masks = preprocess_text(texts, tokenizer)
    model = build_text_model()
    features = model([input_ids, attention_masks])[0][:, 0, :].numpy()
    
    kmeans = load(model_path)
    new_labels = kmeans.predict(features)
    print("Predicted cluster labels for new data:\n", new_labels)

if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    model_path = sys.argv[2]
    main(file_path, model_path)
