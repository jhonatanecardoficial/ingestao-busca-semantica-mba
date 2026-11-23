# Plano de Resolução — Ingestão e Busca Semântica com LangChain e Postgres

**Desafio:** Ingestão e Busca Semântica com LangChain e Postgres (Módulo: Introdução ao LangChain)  
**URL do Desafio:** https://plataforma.fullcycle.com.br/courses/a091b0fe-a5c6-4287-a3d3-1ec61defcfd3/430/224/266/conteudos?projeto=70&fase=293  
**Repositório Template:** https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca  
**Pasta Local Dedicada:** `d:\Ecard\MBA\ingestao-busca-semantica-mba`  
**Repositório GitHub (Entregável):** `https://github.com/jhonatanecardoficial/ingestao-busca-semantica-mba` (público, sufixo `-mba`)

---

## 1. Objetivo

Entregar uma aplicação em Python capaz de:
1. **Ingestão:** Ler um arquivo PDF (`document.pdf`), dividir o texto em chunks e armazenar os embeddings em um banco de dados PostgreSQL com extensão `pgVector`.
2. **Busca e CLI:** Permitir que o usuário faça perguntas via linha de comando (`src/chat.py`) e receba respostas fundamentadas **exclusivamente** no conteúdo do PDF.

---

## 2. Tecnologias Obrigatórias e Pacotes Recomendados

- **Linguagem:** Python
- **Framework:** LangChain
- **Banco de Dados:** PostgreSQL + pgVector (via Docker & Docker Compose)
- **Pacotes recomendados:**
  - Split: `from langchain_text_splitters import RecursiveCharacterTextSplitter`
  - Embeddings (OpenAI): `from langchain_openai import OpenAIEmbeddings`
  - Embeddings (Gemini): `from langchain_google_genai import GoogleGenerativeAIEmbeddings`
  - Leitura de PDF: `from langchain_community.document_loaders import PyPDFLoader`
  - Armazenamento vetorial: `from langchain_postgres import PGVector`
  - Busca vetorial: `similarity_search_with_score(query, k=10)`

> **Observação sobre modelos e dimensões:** Modelos de embeddings diferentes geram vetores com dimensões distintas. Caso o modelo de embedding seja alterado após a primeira ingestão, é necessário apagar a collection existente ou recriar o volume do banco e executar a ingestão do zero.

---

## 3. Estrutura Obrigatória do Projeto

Fazer um fork do repositório: `https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca` (renomeando o fork para `ingestao-busca-semantica-mba`).

```text
ingestao-busca-semantica-mba/
├── docker-compose.yml
├── requirements.txt    # Dependências
├── .env.example        # Template das variáveis de ambiente
├── src/
│   ├── ingest.py       # Script de ingestão do PDF
│   ├── search.py       # Script de busca
│   └── chat.py         # CLI para interação com usuário
├── document.pdf        # PDF para ingestão
└── README.md           # Instruções de execução
```

---

## 4. Requisitos Funcionais Detalhados

### 4.1 Ingestão do PDF (`src/ingest.py`)
- Carregar `document.pdf` usando `PyPDFLoader`.
- Dividir o conteúdo em chunks de **1000 caracteres** com **overlap de 150** usando `RecursiveCharacterTextSplitter`.
- Converter cada chunk em embedding utilizando o provedor selecionado (OpenAI ou Gemini).
- Persistir os vetores no PostgreSQL com `pgVector` via `PGVector`.

### 4.2 Busca Semântica e Prompt (`src/search.py`)
Ao receber uma pergunta:
1. Vetorizar a pergunta.
2. Buscar os **10 resultados mais relevantes (`k=10`)** no banco vetorial usando `similarity_search_with_score(query, k=10)`.
3. Montar o prompt concatenando os resultados no `{resultados concatenados do banco de dados}` e invocar a LLM.
4. Retornar a resposta ao usuário.

**Template de prompt obrigatório:**

```text
CONTEXTO:
{resultados concatenados do banco de dados}

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
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
```

### 4.3 Interação via CLI (`src/chat.py`)
- Simular um chat no terminal seguindo o formato esperado:
  ```text
  Faça sua pergunta:

  PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
  RESPOSTA: O faturamento foi de 10 milhões de reais.
  ```
- Perguntas fora do contexto devem retornar exatamente:
  ```text
  PERGUNTA: Quantos clientes temos em 2024?
  RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
  ```

---

## 5. Etapas de Execução

1. **Bootstrap e GitHub:**
   - Fazer o **fork** oficial do repositório `https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca` na conta `jhonatanecardoficial` já com o nome `ingestao-busca-semantica-mba` (`gh repo fork devfullcycle/mba-ia-desafio-ingestao-busca --fork-name ingestao-busca-semantica-mba`).
   - Clonar/sincronizar o fork dentro de `d:\Ecard\MBA\ingestao-busca-semantica-mba` (preservando este `PLANO.md`) com o remote `origin` apontando para `https://github.com/jhonatanecardoficial/ingestao-busca-semantica-mba`.
2. **Ambiente Virtual e Dependências:**
   - Criar e ativar `venv` (`python -m venv venv`).
   - Revisar e instalar dependências via `requirements.txt`.
   - Configurar `.env.example` e `.env` (sem versionar segredos) com string de conexão PostgreSQL, nome da collection, API Key e modelos de embedding/LLM.
3. **Infraestrutura de Banco de Dados:**
   - Subir o PostgreSQL + pgVector com `docker compose up -d`.
   - Validar que o serviço e a extensão `vector` estão ativos.
4. **Implementação:**
   - Desenvolver `src/ingest.py` com chunking `(1000, 150)` e persistência em `PGVector`.
   - Desenvolver `src/search.py` com busca `k=10` e o prompt estrito definido no desafio.
   - Desenvolver `src/chat.py` com loop interativo no terminal.
5. **Documentação:**
   - Atualizar o `README.md` com requisitos, configuração do `.env` e ordem exata de execução.
6. **Validação:**
   - Subir o banco: `docker compose up -d`
   - Rodar a ingestão: `python src/ingest.py`
   - Rodar o chat: `python src/chat.py`
   - Testar perguntas presentes no `document.pdf` e perguntas fora do contexto para confirmar conformidade total.
7. **Publicação:**
   - Commit e push para `https://github.com/jhonatanecardoficial/ingestao-busca-semantica-mba`.
