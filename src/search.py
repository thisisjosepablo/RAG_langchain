from typing import List, Dict, Any
import time
from src.vectorstore import VectorStore
from src.embedding import EmbeddingManager
from langchain_anthropic import ChatAnthropic


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str,Any]]:
        """
            Retrieve relevant documents for a query

            Args:
                query: The search query
                top_k: Number of top results to return
                score_threshold: Minimum similarity score threshold

            Returns:
                List of dictionaries containing retrieved documents and metadata
        """
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = top_k
            )

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                'id': doc_id,
                                'content': document,
                                'metadata': metadata,
                                'similarity_score': similarity_score,
                                'distance': distance,
                                'rank' : i + 1

                            }
                        )

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

        except Exception as e:
            print(f"Problem searching in vector store: {e}")
            return []
        
        return retrieved_docs


class AdvancedRAGPipeline:
    def __init__(self, retriever: RAGRetriever, llm: ChatAnthropic):
        self.retriever = retriever
        self.llm = llm
        self.history = [] # Save query history
    
    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, stream: bool = False, summarize: bool = False) -> Dict:
        results = self.retriever.retrieve(question, top_k = top_k, score_threshold = min_score)
        if not results:
            print("No context obtained from RAG")
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc["content"] for doc in results])
            sources = [{
                "source" : doc["metadata"].get("source", "unknown"),
                "title" : doc["metadata"].get("title", "unknown"),
                "content_length": doc["metadata"].get("content_length", "unknown"),
                "page_number": f"{doc["metadata"].get("page", "unknown")}/{doc["metadata"].get("total_pages", "unknown")}",
                "score": doc["similarity_score"]
            } for doc in results]

            # Streaming answer simulation
            prompt = f"""
                Use the following context to answer the quesion concisely.

                Contest:
                {context}

                Question:
                {question}
            """

            if stream:
                print("Streaming anser:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i: i+80], end = "", flush = True)
                    time.sleep(0.05)
                print()
            
            response = self.llm.invoke([prompt.format(context = context, question = question)])
            answer = response.content

            # Add citations to answer
            citations = [
                f"({i+1}) File: {source.get("title")} with source {source.get("source")}. Pag {source.get("page")}/{source.get("total_pages")}"
                for i, source in enumerate(sources)
            ]
            answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

            summary = None
            if summarize and answer:
                summary_prompt = f"Summarize the following answer in 2 sentences: \n{answer}"
                summary_resp = self.llm.invoke([summary_prompt])
                summary = summary_resp.content
            
            # Store query history
            self.history.append({
                "question": question,
                "answer": answer,
                "summary": summary,
                "sources" : sources
            })

            return {
                "question": question,
                "answer": answer,
                "summary": summary,
                "sources" : sources,
                "history" : self.history
            }