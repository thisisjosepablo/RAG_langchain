from src.data_loader import load_all_documents
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.search import RAGRetriever, AdvancedRAGPipeline
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv


load_dotenv()

## Example usage

if __name__ == "__main__":
    #docs = load_all_documents("data")
    em = EmbeddingManager()
    vectorstore = VectorStore()

    #chunks = em.split_documents(docs)
    #embeddings = em.generate_embeddings(chunks=chunks)
    #vectorstore.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vector_store=vectorstore,
                            embedding_manager= em)

    llm = ChatAnthropic(
        model = "claude-haiku-4-5",
        temperature = 0.5,
        #max_tokens = 120
    )

    adv_rag = AdvancedRAGPipeline(retriever= retriever, llm=llm)


    result = adv_rag.query("What is the best way to prepare a marathon?", top_k = 3, min_score = 0.1, stream = True, summarize = True)
    print(f"Final answer:\n {result["answer"]}")
    print(f"Summary: {result["summary"]}")
    print(f"Sources: {result["sources"]}")









   