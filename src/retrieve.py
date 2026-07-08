from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone
import os

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
_pc_index = None


def get_index():
    global _pc_index
    if _pc_index is None:
        _pc_index = pc.Index('support-assistant-rag')
    return _pc_index


NAMESPACE = "md-vectors"


def retrieve(query:str, bm25: BM25Okapi,  chunk_ids:list, k:int):
    embedded_query = embed_model.encode(query).tolist()
    index = get_index()
    pc_results = index.query(vector=embedded_query, top_k=k*2, include_metadata=True, namespace=NAMESPACE)
    dense_results = [match.id for match in pc_results.matches]
    chunk_store = {match.id:match.metadata for match in pc_results.matches}

    tokenized_query = re.sub(r'[^\w\s-]', '', query).lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    indices = np.argpartition(bm25_scores, -(k*2))[-(k*2):]
    indices = indices[np.argsort(-bm25_scores[indices])]
    sparse_results = [chunk_ids[i] for i in indices]

    rrf_scores = {}
    for rank, chunk_id in enumerate(dense_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0.0) + (1.0/(60+rank))
    for rank, chunk_id in enumerate(sparse_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0.0) + (1.0/(60+rank))

    final_ranked_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]

    missing_ids = [id for id in final_ranked_ids if id not in chunk_store]
    if missing_ids:
        fetched = index.fetch(ids=missing_ids, namespace=NAMESPACE)
        for fetched_id, vector in fetched.vectors.items():
            chunk_store[fetched_id] = vector.metadata

    retrived_chunks = [chunk_store[id] for id in final_ranked_ids if id in chunk_store]
    return pc_results.matches[0].score if pc_results.matches else 0, retrived_chunks
    
        
    
