# Retrieval Augmented System Using LangChain (NO AI)


Next, we will explain how to use it.
Before cloning and configurating poetry environment, we need to create a `.env` file with the anthropic api key (ANTHROPIC_API_KEY) to be able to use the llm used here. However, you can change to whichever llm you are using.


Then, the only thing you need to do is to follow the `app.py` schema.

The first thing you need to do:
- Load all documents. Actually, we only use pdf files, so all the pdf files you upload to the folder will be processed.
- It is needed to split the documents into chunks, then create embeddings of them and finally saving to the database what we are using, in this case, ChromaDB.


After doing the previous steps, we need to configure the `RAGRetriever` and the `AdvancedRAGPipeline`.


Finally, we could ask a query to the model and then we will get a response. The useful part of this code is that we can get updated and oriented answers according to our requirements, specifically, to the pdfs loaded by ourselves. In addition, the answer will be explained indicating the source from where it was generated, showing parameters as: name of the file, author, number of page, etc.

For example, we can use a LLM trained with data from 1 years ago, but thanks to the pdf obtained by RAG, the answers will be updated with data from nowadays. 