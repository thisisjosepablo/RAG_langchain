import os
import pickle
from pathlib import Path
from src.embedding import EmbeddingManager
import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name:str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    

    def _initialize_store(self):
        try:
            
            # Create persistent ChromaDB client
            path_dir = Path(self.persist_directory).resolve()
            os.makedirs(path_dir, exist_ok = True)
            self.client = chromadb.PersistentClient(path=path_dir)

            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {"description": "PDF document embeddings for RAG"}
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise


    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        for index, (doc, embed) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{index}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = index
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embed.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids = ids,
                embeddings = embeddings_list,
                metadatas = metadatas,
                documents = documents_text
            )

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        
        except Exception as e:
            print(f"Failes to add to the collection {e}")
            raise