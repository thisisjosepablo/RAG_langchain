from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader



def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to langchain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """

    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # PDF files
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(pdf) for pdf in pdf_files]}")
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        loaded = loader.load()
        documents.extend(loaded)


    # Text files


    # CSV files

    # SQL files


    return documents