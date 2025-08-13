import faiss
import numpy as np
from typing import Any, Dict
import os

class FaissStore:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = None
        self.metadata = []
        self.load()

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(384)  # default dim for e5-small-v2

    def add(self, embeddings: np.ndarray, meta: Dict[str, Any]):
        self.index.add(embeddings)
        self.metadata.append(meta)

    def save(self):
        faiss.write_index(self.index, self.index_path)

    def search(self, query_emb: np.ndarray, k: int = 5):
        D, I = self.index.search(query_emb, k)
        return [(self.metadata[i], D[0][idx]) for idx, i in enumerate(I[0])]
