# Django AI Chatbot

A production-grade Django REST API backend with multi-provider LLM support,
conversation memory, fine-tuning, and RAG (Retrieval Augmented Generation).

---

## Features

- Multi-provider AI support (OpenAI, Claude — swap with one env variable)
- Conversation memory with session management
- Fine-tuning support via OpenAI API
- RAG — answer questions from your own documents (PDF, TXT)
- Rate limiting per IP
- Token usage tracking per request
- Django management commands for fine-tuning and document ingestion

---

## Tech Stack

- Python, Django
- OpenAI API, Anthropic Claude API
- ChromaDB (vector database for RAG)
- SQLite (conversation + message storage)

---

## Project Structure

```
CHATBOT_PROJECT/
├── chatbot/
│   ├── ai_providers/
│   │   ├── base.py              # Abstract provider interface
│   │   ├── openai_provider.py   # OpenAI implementation
│   │   ├── claude_provider.py   # Claude implementation
│   │   └── router.py            # Reads LLM_PROVIDER from .env
│   ├── rag/
│   │   ├── document_processor.py  # Chunks documents
│   │   ├── vector_store.py        # ChromaDB store + search
│   │   └── rag_pipeline.py        # Retrieval + generation
│   ├── management/commands/
│   │   ├── finetune.py          # Fine-tuning CLI
│   │   └── ingest_docs.py       # Document ingestion CLI
│   ├── training/data/           # Training data for fine-tuning
│   ├── documents/               # Documents for RAG
│   ├── models.py                # Conversation, ChatMessage, IngestedDocument
│   ├── views.py                 # API endpoints
│   └── urls.py
├── chatbot_project/             # Django settings
├── .env                         # API keys + config
└── manage.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/` | Regular chatbot (GPT-4o) |
| POST | `/rag/` | RAG chatbot (answers from your documents) |
| GET | `/history/<session_id>/` | Conversation history |
| GET | `/rag/documents/` | List ingested documents |

---

## Setup

### 1. Clone and install
```bash
git clone <repo-url>
cd chatbot_project
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=openai            # or "claude"
OPENAI_MODEL=gpt-4o            # or your fine-tuned model
```

### 3. Run migrations
```bash
python manage.py migrate
```

### 4. Start server
```bash
python manage.py runserver
```

---

## Usage

### Regular Chat
```json
POST /chat/
{
    "question": "What is machine learning?",
    "session_id": "user123"
}
```

### RAG Chat (from your documents)
```json
POST /rag/
{
    "question": "What is our leave policy?",
    "session_id": "user123"
}
```

### Ingest a Document
```bash
python manage.py ingest_docs --file chatbot/documents/hr_policy.pdf
python manage.py ingest_docs --list
python manage.py ingest_docs --clear
```

### Fine-Tune a Model
```bash
python manage.py finetune upload --file chatbot/training/data/training_data.jsonl
python manage.py finetune train  --file file-XXXXXXXX
python manage.py finetune status --job-id ftjob-XXXXXXXX
```

### Switch AI Provider
```bash
# .env
LLM_PROVIDER=claude   # switch to Claude
LLM_PROVIDER=openai   # switch to OpenAI
```

---

## Requirements

```
django
openai
anthropic
chromadb
pypdf2
django-ratelimit
python-dotenv
```