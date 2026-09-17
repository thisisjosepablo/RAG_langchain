from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain.agents import create_agent

loader = PyPDFLoader(
    file_path= Path('/home/thisisjosepablo/Escritorio/Agents/langchain/data/raw_pdf/book1.pdf'),
    mode= "page" #"single"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_running_books"
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@tool
def buscar_conocimiento_deportivo(pregunta: str) -> str:
    """Busca en libros de running, gimnasio y nutrición información relevante para responder la pregunta del usuario."""
    resultados = retriever.invoke(pregunta)
    return "\n\n".join(r.page_content for r in resultados)


resultados = retriever.invoke("How to structure series of 10km?")
for r in resultados:
    print("---")
    print(r.page_content[:300])
    print(r.metadata)

# agente = create_agent(
#     model="claude-haiku-4-5-20251001",  # o el modelo que uses
#     tools=[buscar_conocimiento_deportivo],
#     system_prompt="Eres un entrenador personal experto en running, gimnasio y nutrición. "
#                    "Usa la herramienta de búsqueda para fundamentar tus respuestas en los libros disponibles. "
#                    "Cuando no tengas info en los libros, dilo claramente en vez de inventar."
# )