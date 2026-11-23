# Documentação dos Testes Automatizados — Ingestão e Busca Semântica

Este documento detalha a suíte de testes automatizados implementada para validar a integridade, conformidade arquitetural e regras de negócio do desafio técnico.

---

## 📁 Localização da Suíte de Testes

- Arquivo: [`tests/test_rag.py`](tests/test_rag.py)
- Framework: `unittest` (nativo do Python, sem necessidade de dependências extras de runner)

---

## 🎯 Cenários Cobertos

| ID | Teste | Objetivo | Critério de Aceite |
|---|---|---|---|
| **01** | `test_01_pdf_loader_and_chunking` | Validar carregamento do `document.pdf` e regras de chunking | Carregar 34 páginas, gerar 67 chunks com `chunk_size=1000` e `chunk_overlap=150`, contendo os dados da `SuperTechIABrazil`. |
| **02** | `test_02_database_connection` | Validar configuração e protocolo do PostgreSQL | String de conexão formatada com o driver `postgresql+psycopg://`. |
| **03** | `test_03_pgvector_ingestion_and_similarity_search` | Validar persistência no pgVector e busca dos k=10 mais relevantes | Inserir chunks em collection de teste e recuperar exatamente `k=10` documentos com seus respectivos scores de similaridade. |
| **04** | `test_04_prompt_template_strict_rules` | Validar conformidade integral do prompt com as regras do desafio | Conter regras de resposta exclusiva pelo contexto, proibição de conhecimento externo/opiniões e exemplos literais de recusa. |
| **05** | `test_05_rag_chain_in_context_response` | Validar resposta para pergunta presente no documento | Resposta correta retornada: `"O faturamento foi de 10 milhões de reais."`. |
| **06** | `test_06_rag_chain_out_of_context_refusal` | Validar recusa estrita para pergunta fora do contexto | Recusa exata retornada: `"Não tenho informações necessárias para responder sua pergunta."`. |
| **07** | `test_07_live_llm_if_key_configured` | Validar conexão ao vivo com LLM real (OpenAI ou Gemini) | Ativa automaticamente assim que `GOOGLE_API_KEY` ou `OPENAI_API_KEY` for informada no `.env`. |

---

## 🚀 Como Executar os Testes

Com o ambiente virtual ativado e o container PostgreSQL rodando (`docker compose up -d`):

```bash
python -m unittest tests/test_rag.py -v
```

---

## 📊 Evidência de Execução

```text
test_01_pdf_loader_and_chunking (tests.test_rag.TestRAGIngestaoBusca.test_01_pdf_loader_and_chunking)
Valida carregamento do PDF e parâmetros obrigatórios de chunking (1000/150). ... ok
test_02_database_connection (tests.test_rag.TestRAGIngestaoBusca.test_02_database_connection)
Valida se a URL de conexão com o PostgreSQL está formatada corretamente com psycopg. ... ok
test_03_pgvector_ingestion_and_similarity_search (tests.test_rag.TestRAGIngestaoBusca.test_03_pgvector_ingestion_and_similarity_search)
Valida ingestão no pgVector e busca dos k=10 resultados mais relevantes. ... ok
test_04_prompt_template_strict_rules (tests.test_rag.TestRAGIngestaoBusca.test_04_prompt_template_strict_rules)
Valida se o template de prompt contém todas as regras estritas e exemplos de recusa. ... ok
test_05_rag_chain_in_context_response (tests.test_rag.TestRAGIngestaoBusca.test_05_rag_chain_in_context_response)
Valida que a chain responde perguntas dentro do contexto. ... ok
test_06_rag_chain_out_of_context_refusal (tests.test_rag.TestRAGIngestaoBusca.test_06_rag_chain_out_of_context_refusal)
Valida recusa para perguntas fora do contexto. ... ok
test_07_live_llm_if_key_configured (tests.test_rag.TestRAGIngestaoBusca.test_07_live_llm_if_key_configured)
Se GOOGLE_API_KEY ou OPENAI_API_KEY estiver configurada, executa teste real com a LLM. ... skipped 'Nenhuma API Key (OpenAI ou Google) configurada no .env. Pulando teste com LLM ao vivo.'

----------------------------------------------------------------------
Ran 7 tests in 1.137s

OK (skipped=1)
```
