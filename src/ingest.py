import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

# Garantir import de search.py
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from search import get_embeddings, get_database_url, get_collection_name

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")


def ingest_pdf():
    # Resolver caminho do PDF
    pdf_path = Path(PDF_PATH)
    if not pdf_path.is_absolute():
        project_root = current_dir.parent
        candidate = project_root / PDF_PATH
        if candidate.exists():
            pdf_path = candidate
        elif not pdf_path.exists():
            raise FileNotFoundError(f"Arquivo PDF não encontrado em: {PDF_PATH}")

    print(f"Carregando PDF: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    print(f"Total de páginas carregadas: {len(docs)}")

    print("Dividindo texto em chunks (chunk_size=1000, chunk_overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Total de chunks gerados: {len(chunks)}")

    print("Inicializando embeddings e conexão com PostgreSQL...")
    embeddings = get_embeddings()
    connection = get_database_url()
    collection_name = get_collection_name()

    print(f"Armazenando vetores no pgVector (collection: '{collection_name}')...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=connection,
        collection_name=collection_name,
        use_jsonb=True,
        pre_delete_collection=True
    )
    print("Ingestão concluída com sucesso no PostgreSQL com pgVector!")


if __name__ == "__main__":
    ingest_pdf()
