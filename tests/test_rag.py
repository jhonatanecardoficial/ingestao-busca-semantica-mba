import os
import sys
import unittest
from pathlib import Path
from dotenv import load_dotenv

# Adicionar raiz do projeto e pasta src ao sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake import FakeListLLM

from search import (
    PROMPT_TEMPLATE,
    RAGSearchChain,
    get_database_url,
    get_embeddings,
    get_llm,
)


class TestRAGIngestaoBusca(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(PROJECT_ROOT / ".env")
        cls.pdf_path = PROJECT_ROOT / "document.pdf"
        cls.db_url = get_database_url()
        cls.test_embeddings = FakeEmbeddings(size=128)
        cls.test_collection = "test_evaluation_collection"

    def test_01_pdf_loader_and_chunking(self):
        """Valida carregamento do PDF e parâmetros obrigatórios de chunking (1000/150)."""
        self.assertTrue(self.pdf_path.exists(), "document.pdf deve existir na raiz do projeto.")
        
        loader = PyPDFLoader(str(self.pdf_path))
        docs = loader.load()
        self.assertGreater(len(docs), 0, "PDF deve conter páginas válidas.")
        self.assertEqual(len(docs), 34, "PDF deve conter exatamente 34 páginas.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        self.assertEqual(len(chunks), 67, "Divisão do PDF deve gerar 67 chunks.")

        # Verificar se o termo-chave do desafio está presente em algum chunk
        encontrou_empresa = any("SuperTechIABrazil" in c.page_content for c in chunks)
        self.assertTrue(encontrou_empresa, "Os chunks devem conter menção à 'SuperTechIABrazil'.")

    def test_02_database_connection(self):
        """Valida se a URL de conexão com o PostgreSQL está formatada corretamente com psycopg."""
        self.assertTrue(
            self.db_url.startswith("postgresql+psycopg://"),
            f"A URL de conexão deve usar o driver postgresql+psycopg. Atual: {self.db_url}"
        )

    def test_03_pgvector_ingestion_and_similarity_search(self):
        """Valida ingestão no pgVector e busca dos k=10 resultados mais relevantes."""
        loader = PyPDFLoader(str(self.pdf_path))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        vector_store = PGVector.from_documents(
            documents=chunks,
            embedding=self.test_embeddings,
            connection=self.db_url,
            collection_name=self.test_collection,
            use_jsonb=True,
            pre_delete_collection=True
        )

        pergunta = "Qual o faturamento da Empresa SuperTechIABrazil?"
        results = vector_store.similarity_search_with_score(pergunta, k=10)

        self.assertEqual(len(results), 10, "A busca vetorial deve retornar exatamente k=10 chunks.")
        for doc, score in results:
            self.assertIsNotNone(doc.page_content)
            self.assertIsInstance(score, float)

    def test_04_prompt_template_strict_rules(self):
        """Valida se o template de prompt contém todas as regras estritas e exemplos de recusa."""
        regras_obrigatorias = [
            "Responda somente com base no CONTEXTO.",
            'Se a informação não estiver explicitamente no CONTEXTO, responda:',
            '"Não tenho informações necessárias para responder sua pergunta."',
            "Nunca invente ou use conhecimento externo.",
            "Nunca produza opiniões ou interpretações além do que está escrito.",
            'Pergunta: "Qual é a capital da França?"',
            'Pergunta: "Quantos clientes temos em 2024?"',
            'Pergunta: "Você acha isso bom ou ruim?"'
        ]

        for regra in regras_obrigatorias:
            self.assertIn(regra, PROMPT_TEMPLATE, f"Regra ausente no prompt: '{regra}'")

    def test_05_rag_chain_in_context_response(self):
        """Valida que a chain responde perguntas dentro do contexto."""
        vector_store = PGVector(
            embeddings=self.test_embeddings,
            collection_name=self.test_collection,
            connection=self.db_url,
            use_jsonb=True
        )

        resposta_esperada = "O faturamento foi de 10 milhões de reais."
        mock_llm = FakeListLLM(responses=[resposta_esperada])
        chain = RAGSearchChain(vector_store, mock_llm, PROMPT_TEMPLATE)

        resposta = chain.invoke({"pergunta": "Qual o faturamento da Empresa SuperTechIABrazil?"})
        self.assertEqual(resposta, resposta_esperada)

    def test_06_rag_chain_out_of_context_refusal(self):
        """Valida recusa para perguntas fora do contexto."""
        vector_store = PGVector(
            embeddings=self.test_embeddings,
            collection_name=self.test_collection,
            connection=self.db_url,
            use_jsonb=True
        )

        recusa_esperada = "Não tenho informações necessárias para responder sua pergunta."
        mock_llm = FakeListLLM(responses=[recusa_esperada])
        chain = RAGSearchChain(vector_store, mock_llm, PROMPT_TEMPLATE)

        resposta = chain.invoke({"pergunta": "Quantos clientes temos em 2024?"})
        self.assertEqual(resposta, recusa_esperada)

    def test_07_live_llm_if_key_configured(self):
        """Se GOOGLE_API_KEY ou OPENAI_API_KEY estiver configurada, executa teste real com a LLM."""
        load_dotenv(PROJECT_ROOT / ".env")
        has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
        has_google = bool(os.getenv("GOOGLE_API_KEY", "").strip())

        if not (has_openai or has_google):
            self.skipTest("Nenhuma API Key (OpenAI ou Google) configurada no .env. Pulando teste com LLM ao vivo.")

        try:
            embeddings = get_embeddings()
            llm = get_llm()
            self.assertIsNotNone(embeddings)
            self.assertIsNotNone(llm)
            print(f"\n[LIVE TEST] Provedor configurado e validado com sucesso!")
        except Exception as e:
            self.fail(f"Falha ao instanciar provedor com a API Key configurada: {e}")


if __name__ == "__main__":
    unittest.main()
