# 🎬 Advanced MovieBot

An intelligent movie assistant chatbot powered by a local LLM and real-world data from **Wikipedia**.

---

## Objective

Advanced MovieBot is a RAG-based assistant that uses real movie plot data to answer user questions conversationally through a FastAPI REST API.

The dataset is available on HuggingFace [Wikipedia Movie Plots dataset](https://huggingface.co/datasets/vishnupriyavr/wiki-movie-plots-with-summaries)

It demonstrates how Retrieval-Augmented Generation (RAG) can be used with local LLMs to build intelligent assistants. Similar techniques can be applied to build customer service bots or domain-specific AI helpers.

---

## Implementation

This assistant is built using:

- **LangChain** – for building RAG pipelines and memory handling
- **FastAPI** – for REST API server with embedded HTML chat interface
- **Ollama/vLLM** – for running local or server-based LLMs (Note: vLLM is not supported on Mac!!)
- **Gemma 2 (9B)** – a local open-source LLM for generating responses
- **FAISS** – for fast vector search and retrieval of relevant documents
- **HuggingFace Embeddings** – to convert texts into vector representations
- **Session Memory** – preserves conversation history for contextual continuity

The dataset is preprocessed from Wikipedia's movie plot summaries, chunked, embedded, and indexed locally using FAISS for efficient retrieval.

---

## Running

### Prerequisites

- Python 3.9+
- Ollama (Mac) or vLLM (Linux/Windows with GPU)

### Setup

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/elhambbi/Assistant_MovieBot.git
   cd Assistant_MovieBot
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure and run the server**

   If using Ollama (default):
   ```bash
   # Install Ollama and download Gemma 2 model, See Ollama section below
   ollama pull gemma2:9b
   ollama serve
   export LLM_BACKEND=ollama
   export MODEL_NAME=gemma2:9b
   ```
   If using vLLM:
   ```bash
   export LLM_BACKEND=vllm
   export VLLM_MODEL_HF=google/gemma-2-9b
   ```
   Then run the server:
   ```bash
   uvicorn src.app:app --port 8000 --reload
   ```

Falls back to Ollama if vLLM is unavailable.

4. **Open your browser and start chatting!**
   ```
   http://localhost:8000
   ```

*Note:* Creating the FAISS index for document retrieval takes 10-15 minutes during the first run. It will be saved and loaded automatically in subsequent runs.

---
## Example Output

![example chat](example_chat.png)

---
## API Endpoints

### GET /
Returns the chat interface.

### POST /query
Submit a movie question.

**Request:**
```json
{
  "question": "What is the genre of Funny people?",
  "session_id": "user_123"
}
```

**Response:**
```json
{
  "answer": "Funny people is a comedy movie...",
  "session_id": "user_123",
  "history": [...]
}
```

### GET /health
Health check endpoint.

---

## Ollama

### Installation

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -sSL https://ollama.com/install.sh | bash
```

### Downloading Models

To download a specific model (list available at [ollama.com/search](https://ollama.com/search)):

```bash
ollama pull gemma2:9b
```

Models are saved in `~/.ollama`

### List Downloaded Models

```bash
ollama list
```

If empty on Mac, try:
```bash
brew services restart ollama
ollama list
```

### Testing Models

```bash
ollama run gemma2:9b "Hello World!"
```

### Removing Models

```bash
ollama rm <model-name>
```
---