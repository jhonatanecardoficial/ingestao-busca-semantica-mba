# Ingestão e Busca Semântica com LangChain e PostgreSQL (pgVector)

Desafio técnico do **MBA em Engenharia de Software com IA** — Full Cycle.  
Módulo: **Introdução ao LangChain**

Repositório: [https://github.com/jhonatanecardoficial/ingestao-busca-semantica-mba](https://github.com/jhonatanecardoficial/ingestao-busca-semantica-mba)

---

## 🎯 Objetivo

Implementar uma aplicação RAG (*Retrieval-Augmented Generation*) em Python utilizando o framework **LangChain** e o banco de dados **PostgreSQL** com a extensão **pgVector**, capaz de:

1. **Ingestão**: Carregar um arquivo PDF (`document.pdf`), dividi-lo em chunks padronizados, gerar embeddings e persistir os vetores no PostgreSQL com `pgVector`.
2. **Busca Semântica & CLI**: Permitir que o usuário faça perguntas no terminal e receba respostas baseadas **estritamente** no contexto do documento, recusando responder qualquer pergunta com informação ausente.

---

## 🏗️ Estrutura do Projeto

```text
ingestao-busca-semantica-mba/
├── docker-compose.yml   # Definição do serviço PostgreSQL 17 + extensão pgVector
├── requirements.txt     # Dependências do projeto (LangChain, pgvector, psycopg, etc.)
├── .env.example         # Template com todas as variáveis de ambiente necessárias
├── document.pdf         # Arquivo PDF base para ingestão
├── README.md            # Documentação e instruções de execução
├── TESTES.md            # Documentação e evidências da suíte de testes
├── src/
│   ├── ingest.py        # Pipeline de carregamento, chunking e gravação vetorial
│   ├── search.py        # Busca semântica (k=10), montagem do prompt e chamada LLM
│   └── chat.py          # Interface de chat interativa via terminal (CLI)
└── tests/
    └── test_rag.py      # Suíte de testes automatizados
```

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python (versão 3.10 ou superior)
- **Framework RAG**: LangChain (`langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-postgres`)
- **Provedores de IA Suportados**:
  - **OpenAI**: `OpenAIEmbeddings` (`text-embedding-3-small`) e `ChatOpenAI` (`gpt-4o-mini`)
  - **Google Gemini**: `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`) e `ChatGoogleGenerativeAI` (`gemini-2.5-flash`)
- **Banco de Dados Vetorial**: PostgreSQL 17 com extensão `pgVector` (`pgvector/pgvector:pg17`)
- **Containerização**: Docker e Docker Compose

---

## 📋 Pré-requisitos

Antes de iniciar, certifique-se de ter instalado em sua máquina:

- [Git](https://git-scm.com/)
- [Python 3.10+](https://www.python.org/)
- [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/)
- Uma chave de API da **OpenAI** ou do **Google Gemini**

---

## 🚀 Guia Passo a Passo de Execução

### 1. Clonar o Repositório

```bash
git clone https://github.com/jhonatanecardoficial/ingestao-busca-semantica-mba.git
cd ingestao-busca-semantica-mba
```

---

### 2. Criar e Ativar o Ambiente Virtual

No **Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

No **Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 3. Instalar as Dependências

```bash
pip install -r requirements.txt
```

---

### 4. Configurar as Variáveis de Ambiente (`.env`)

Copie o arquivo de exemplo para criar seu `.env`:

```bash
cp .env.example .env
```
*(No Windows PowerShell: `Copy-Item .env.example .env`)*

Abra o arquivo `.env` e configure conforme o provedor que deseja utilizar:

#### Opção A: Usando OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sua-chave-openai-aqui
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
OPENAI_CHAT_MODEL='gpt-4o-mini'

POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=pdf_documents
PDF_PATH=document.pdf
```

#### Opção B: Usando Google Gemini
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=sua-chave-google-aqui
GOOGLE_EMBEDDING_MODEL='models/gemini-embedding-001'
GOOGLE_CHAT_MODEL='gemini-2.5-flash'

POSTGRES_PORT=5432
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=pdf_documents
PDF_PATH=document.pdf
```

> **Nota sobre portas no Windows**: Caso você já possua um PostgreSQL instalado localmente na porta `5432`, altere `POSTGRES_PORT=5435` e ajuste o `DATABASE_URL` para `postgresql+psycopg://postgres:postgres@localhost:5435/rag`.

---

### 5. Subir o Banco de Dados com pgVector

Inicie os containers com o Docker Compose:

```bash
docker compose up -d
```

Verifique se o container está rodando e saudável:
```bash
docker compose ps
```

---

### 6. Executar a Ingestão do PDF

Execute o script de ingestão:

```bash
python src/ingest.py
```

O script irá:
1. Carregar as 34 páginas do `document.pdf`.
2. Dividir o texto em chunks de **1000 caracteres** com **overlap de 150 caracteres** (`RecursiveCharacterTextSplitter`).
3. Gerar os vetores de embedding para cada chunk.
4. Salvar os vetores na collection configurada no PostgreSQL com `pgVector`.

Saída esperada:
```text
Carregando PDF: document.pdf
Total de páginas carregadas: 34
Dividindo texto em chunks (chunk_size=1000, chunk_overlap=150)...
Total de chunks gerados: 67
Inicializando embeddings e conexão com PostgreSQL...
Armazenando vetores no pgVector (collection: 'pdf_documents')...
Ingestão concluída com sucesso no PostgreSQL com pgVector!
```

---

### 7. Interagir via Chat CLI

Inicie a interface de terminal:

```bash
python src/chat.py
```

Exemplos de interação:

#### Pergunta dentro do contexto:
```text
Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento da Empresa SuperTechIABrazil foi de R$ 10.000.000,00 (10 milhões de reais).
```

#### Perguntas fora do contexto (deve recusar conforme regra):
```text
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: Você acha isso bom ou ruim?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Para encerrar o chat, digite `sair`, `exit` ou pressione `Ctrl+C`.

---

## 🔍 Teste Direto de Busca

Você também pode testar uma pergunta diretamente via `src/search.py`:

```bash
python src/search.py "Qual o faturamento da Empresa SuperTechIABrazil?"
```

---

## 📌 Requisitos Atendidos

| Requisito | Status | Implementação |
|---|---|---|
| Split com chunks de 1000 e overlap 150 | ✅ Concluído | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` em `src/ingest.py` |
| Geração e armazenamento de embeddings | ✅ Concluído | `PGVector.from_documents` com OpenAI ou Gemini |
| Banco de dados PostgreSQL + pgVector | ✅ Concluído | `docker-compose.yml` (`pgvector/pgvector:pg17`) |
| Busca vetorial dos 10 mais relevantes (`k=10`) | ✅ Concluído | `similarity_search_with_score(query, k=10)` em `src/search.py` |
| Prompt e regras estritas de recusa | ✅ Concluído | Template com regras inegociáveis e exemplos explícitos de recusa |
| Interface CLI interativa | ✅ Concluído | `src/chat.py` com loop no terminal e saída formatada |
| Repositório público com sufixo `-mba` | ✅ Concluído | Fork oficial em `jhonatanecardoficial/ingestao-busca-semantica-mba` |
