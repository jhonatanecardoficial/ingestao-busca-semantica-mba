import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

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


def get_embeddings():
    load_dotenv()
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

    if provider in ["openai"]:
        if not openai_key:
            raise ValueError("OPENAI_API_KEY não foi configurada no arquivo .env.")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip("'\"")
        return OpenAIEmbeddings(model=model, openai_api_key=openai_key)
    elif provider in ["gemini", "google"]:
        if not google_key:
            raise ValueError("GOOGLE_API_KEY não foi configurada no arquivo .env.")
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004").strip("'\"")
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=google_key)
    else:
        raise ValueError(f"Provedor '{provider}' não suportado. Use 'openai' ou 'gemini'.")


def get_llm():
    load_dotenv()
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

    if provider in ["openai"]:
        if not openai_key:
            raise ValueError("OPENAI_API_KEY não foi configurada no arquivo .env.")
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip("'\"")
        return ChatOpenAI(model=model, temperature=0, openai_api_key=openai_key)
    elif provider in ["gemini", "google"]:
        if not google_key:
            raise ValueError("GOOGLE_API_KEY não foi configurada no arquivo .env.")
        model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash").strip("'\"")
        return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=google_key)
    else:
        raise ValueError(f"Provedor '{provider}' não suportado. Use 'openai' ou 'gemini'.")


def get_database_url():
    load_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        port = os.getenv("POSTGRES_PORT", "5432").strip()
        url = f"postgresql+psycopg://postgres:postgres@localhost:{port}/rag"
    else:
        if url.startswith("postgresql://") and not url.startswith("postgresql+psycopg://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_collection_name():
    load_dotenv()
    return os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents").strip("'\"") or "pdf_documents"


def get_vector_store():
    embeddings = get_embeddings()
    connection = get_database_url()
    collection_name = get_collection_name()
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )


class RAGSearchChain:
    def __init__(self, vector_store, llm, prompt_template):
        self.vector_store = vector_store
        self.llm = llm
        self.prompt_template = prompt_template

    def invoke(self, input_data):
        if isinstance(input_data, dict):
            question = input_data.get("pergunta") or input_data.get("question") or ""
        else:
            question = str(input_data)

        question = question.strip()
        if not question:
            return "Por favor, forneça uma pergunta válida."

        # 1. Vetorizar a pergunta e buscar os 10 resultados mais relevantes (k=10)
        results = self.vector_store.similarity_search_with_score(question, k=10)

        # 2. Concatenar os resultados recuperados
        context_parts = []
        for doc, _score in results:
            content = doc.page_content.strip()
            if content:
                context_parts.append(content)
        contexto = "\n\n".join(context_parts)

        # 3. Montar o prompt
        prompt_text = self.prompt_template.format(
            contexto=contexto,
            pergunta=question
        )

        # 4. Chamar a LLM e retornar a resposta
        response = self.llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer.strip()

    def __call__(self, input_data):
        return self.invoke(input_data)


def search_prompt(question=None):
    vector_store = get_vector_store()
    llm = get_llm()
    chain = RAGSearchChain(vector_store, llm, PROMPT_TEMPLATE)
    if question is not None:
        return chain.invoke(question)
    return chain


if __name__ == "__main__":
    pergunta_teste = sys.argv[1] if len(sys.argv) > 1 else "Qual o faturamento da Empresa SuperTechIABrazil?"
    print(f"PERGUNTA: {pergunta_teste}")
    resposta = search_prompt(pergunta_teste)
    print(f"RESPOSTA: {resposta}")
