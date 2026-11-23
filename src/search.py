"""Módulo de busca semântica e geração de respostas via RAG com PGVector e LangChain."""

import os
import sys
from typing import Any, Optional, Union

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _resolve_provider() -> tuple[str, str]:
    """Resolve o provedor de IA ativo ('openai' ou 'gemini') e sua respectiva API Key."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not provider:
        if openai_key:
            provider = "openai"
        elif google_key:
            provider = "gemini"
        else:
            raise ValueError(
                "Nenhuma chave de API configurada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env."
            )

    if provider == "openai":
        if not openai_key:
            raise ValueError("OPENAI_API_KEY não foi configurada no arquivo .env.")
        return "openai", openai_key

    if provider in ("gemini", "google"):
        if not google_key:
            raise ValueError("GOOGLE_API_KEY não foi configurada no arquivo .env.")
        return "gemini", google_key

    raise ValueError(f"Provedor '{provider}' não suportado. Use 'openai' ou 'gemini'.")


def get_embeddings() -> Embeddings:
    """Instancia o modelo de embeddings conforme o provedor configurado."""
    provider, api_key = _resolve_provider()

    if provider == "openai":
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip("'\"")
        return OpenAIEmbeddings(model=model, openai_api_key=api_key)

    model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001").strip("'\"")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def get_llm() -> BaseChatModel:
    """Instancia o modelo de chat (LLM) com temperatura zero para respostas determinísticas."""
    provider, api_key = _resolve_provider()

    if provider == "openai":
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip("'\"")
        return ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)

    model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash").strip("'\"")
    return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)


def get_database_url() -> str:
    """Retorna a URL de conexão PostgreSQL normalizada para o driver psycopg3."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        port = os.getenv("POSTGRES_PORT", "5432").strip()
        return f"postgresql+psycopg://postgres:postgres@localhost:{port}/rag"

    if url.startswith("postgresql://") and not url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_collection_name() -> str:
    """Retorna o nome da collection utilizada no PGVector."""
    return os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents").strip("'\"") or "pdf_documents"


def get_vector_store() -> PGVector:
    """Cria a instância de conexão com o repositório vetorial PGVector."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=get_collection_name(),
        connection=get_database_url(),
        use_jsonb=True,
    )


class RAGSearchChain:
    """Encadeamento RAG para busca vetorial (k=10) e resposta restrita ao contexto."""

    def __init__(self, vector_store: PGVector, llm: BaseChatModel, prompt_template: str) -> None:
        """Inicializa a chain com o vector store, a LLM e o template de prompt."""
        self.vector_store = vector_store
        self.llm = llm
        self.prompt_template = prompt_template

    def invoke(self, input_data: Union[dict[str, Any], str]) -> str:
        """Vetoriza a pergunta, recupera os k=10 chunks mais próximos e consulta a LLM."""
        if isinstance(input_data, dict):
            question = str(input_data.get("pergunta") or input_data.get("question") or "")
        else:
            question = str(input_data)

        question = question.strip()
        if not question:
            return "Por favor, forneça uma pergunta válida."

        results = self.vector_store.similarity_search_with_score(question, k=10)

        context_parts = [doc.page_content.strip() for doc, _score in results if doc.page_content.strip()]
        contexto = "\n\n".join(context_parts)

        prompt_text = self.prompt_template.format(
            contexto=contexto,
            pergunta=question,
        )

        response = self.llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer.strip()

    def __call__(self, input_data: Union[dict[str, Any], str]) -> str:
        """Permite invocar a chain diretamente como função."""
        return self.invoke(input_data)


def search_prompt(question: Optional[str] = None) -> Union[RAGSearchChain, str]:
    """Retorna a chain RAG inicializada ou responde diretamente caso uma pergunta seja passada."""
    chain = RAGSearchChain(get_vector_store(), get_llm(), PROMPT_TEMPLATE)
    if question is not None:
        return chain.invoke(question)
    return chain


if __name__ == "__main__":
    pergunta_teste = sys.argv[1] if len(sys.argv) > 1 else "Qual o faturamento da Empresa SuperTechIABrazil?"
    print(f"PERGUNTA: {pergunta_teste}")
    resposta = search_prompt(pergunta_teste)
    print(f"RESPOSTA: {resposta}")
