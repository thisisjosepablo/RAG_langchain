from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents



class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""

    def __init__(self, model_name:str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
            Initialize the embedding manager

            Args:
                model_name
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loades successfully. Embedding dimension: {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def split_documents(self, documents : List[Any]) -> List[Any]:
        """Split documents into smaller chunks for better RAG performance"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""]
        )

        split_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        return split_docs

    
    def generate_embeddings(self, chunks: List[Any] ) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        if isinstance(chunks[0], str):
            texts = chunks
        else:
            texts = [chunk.page_content for chunk in chunks]

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar = True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.get_embedding_dimension()
