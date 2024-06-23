from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np
from rawSearch import TFIDF
import faiss
tf_idf = TFIDF()

# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

def get_embedding(item):
    tokens = tokenizer(item["text"],max_length=128, return_tensors='pt', padding="max_length", truncation=True)
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()  
    return {"embedding": embeddings.tolist()}


class ReRanker():
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")


    def rank(self, query, docs):
        query_embedding = get_embedding({"text": query})["embedding"]
        ds = Dataset.from_dict({'text': [doc for doc in docs]})
        ds = ds.map(get_embedding, batched=True, batch_size=128)
        ds=ds.add_faiss_index(column="embedding")
        ds = ds.to_pandas()
        # Thực hiện tìm vector tương đồng với query
        doc_embeddings = np.vstack(ds['embedding'])
        scores = cosine_similarity(query_embedding, doc_embeddings).flatten()
        sorted_indices = scores.argsort()[::-1]
        return [(docs[i], scores[i]) for i in sorted_indices]