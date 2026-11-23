"""Script de ingestão de documentos PDF no PostgreSQL com extensão pgVector."""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Garantir import de search.py independentemente do diretório de execução
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from search import get_collection_name, get_database_url, get_embeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")


def _resolve_pdf_path(raw_path: str) -> Path:
    """Resolve o caminho do PDF buscando no caminho direto ou na raiz do projeto."""
    pdf_path = Path(raw_path)
    if pdf_path.is_absolute():
        if not pdf_path.exists():
            raise FileNotFoundError(f"Arquivo PDF não encontrado em: {raw_path}")
        return pdf_path

    candidate = current_dir.parent / raw_path
    if candidate.exists():
        return candidate
    if pdf_path.exists():
        return pdf_path

    raise FileNotFoundError(f"Arquivo PDF não encontrado em: {raw_path}")


def _add_batch_with_retry(vector_store: PGVector, batch: list, max_attempts: int = 5) -> None:
    """Insere um lote de documentos no PGVector com backoff para erros de rate limit (429)."""
    for attempt in range(max_attempts):
        try:
            vector_store.add_documents(batch)
            return
        except Exception as exc:
            err_msg = str(exc)
            is_rate_limit = "429" in err_msg or "Quota exceeded" in err_msg or "ResourceExhausted" in err_msg
            if is_rate_limit and attempt < max_attempts - 1:
                wait_time = 25 * (attempt + 1)
                print(
                    f"    Limite de requisições por minuto atingido (429). "
                    f"Aguardando {wait_time}s para retomar..."
                )
                time.sleep(wait_time)
            else:
                raise


def ingest_pdf() -> None:
    """Carrega o PDF, divide em chunks (1000/150) e persiste os embeddings no pgVector."""
    pdf_path = _resolve_pdf_path(PDF_PATH)

    print(f"Carregando PDF: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    print(f"Total de páginas carregadas: {len(docs)}")

    print("Dividindo texto em chunks (chunk_size=1000, chunk_overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
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
        pre_delete_collection=True,
    )

    batch_size = 15
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"Inserindo {len(chunks)} chunks em {total_batches} lotes de até {batch_size} chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Processando lote {batch_num}/{total_batches} ({len(batch)} chunks)...")
        _add_batch_with_retry(vector_store, batch)

    print("Ingestão concluída com sucesso no PostgreSQL com pgVector!")


if __name__ == "__main__":
    ingest_pdf()
