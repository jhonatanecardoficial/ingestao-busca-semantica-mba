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
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
        pre_delete_collection=True
    )

    import time
    batch_size = 15
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"Inserindo {len(chunks)} chunks em {total_batches} lotes de até {batch_size} chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Processando lote {batch_num}/{total_batches} ({len(batch)} chunks)...")

        for attempt in range(5):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Quota exceeded" in err_msg or "ResourceExhausted" in err_msg:
                    wait_time = 25 * (attempt + 1)
                    print(f"    Limite de requisições por minuto atingido (429). Aguardando {wait_time}s para retomar...")
                    time.sleep(wait_time)
                else:
                    raise e

    print("Ingestão concluída com sucesso no PostgreSQL com pgVector!")


if __name__ == "__main__":
    ingest_pdf()
