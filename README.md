# GPO 2018 Annual Report — Q&A (Baseline RAG)

Use Case 1 of the RAG hands-on exercises: a question-answering chatbot over the
U.S. Government Publishing Office 2018 Annual Report (82 pages). Built with LangChain +
Google Gemini on the same structure as `chatbot-pilot`, with both a CLI and a browser UI.

This is the "normal" document: ordinary prose on two-column pages, with numbers written
into the sentences. It is the baseline the other two use cases are compared against.

## How it works

```
Offline (once):  PDF -> Reading order -> Sentence chunks -> local bge-small embeddings (+ BM25) -> .index/
Every question:  Question -> Hybrid search (dense + BM25, RRF) -> Gemini rerank -> Cited answer
```

| Problem with basic RAG | Fix |
|---|---|
| Two columns come out interleaved | Read band by band: left column, then right; drop the running footer |
| Chunks cut mid-sentence | Whole sentences only, ~450 tokens, ~60 tokens of overlap |
| Right passage found, but ranked low | Hybrid search (meaning + exact words, Reciprocal Rank Fusion), then Gemini reranks the top 20 down to 5 |
| Model fills gaps from memory | Numbered passages with page numbers; the model must cite them or reply "I could not find this in the document." |

## Project structure

```
ingest.py       # PDF -> chunks: reading order + sentence chunks
chatbot.py      # embeddings, vector store, hybrid search, rerank, rag_chat()
server.py       # FastAPI bridge exposing rag_chat() to the browser UI
static/         # HTML/CSS/JS frontend (index.html, style.css, script.js)
data/           # GPO-Anual-report.pdf, the source document
pyproject.toml  # dependencies (managed with uv)
```

The first run chunks and embeds the PDF and caches the result in `.index/`. Later runs
load it from there. Delete `.index/` to rebuild.

Embeddings run locally with [fastembed](https://github.com/qdrant/fastembed)
(`BAAI/bge-small-en-v1.5`, CPU, no API quota). The model (~67 MB) downloads once on
first run. Gemini is used only for answering and reranking.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) installed
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Setup

```bash
uv sync
```

## Running it

### Option A — command-line chatbot

```bash
uv run python chatbot.py
```

You'll be prompted once for your Gemini API key (input is hidden, never saved to
disk). Chat in the terminal; type `quit` to exit.

### Option B — browser UI

```bash
uv run uvicorn server:app --reload
```

Then open **http://127.0.0.1:8000/**. The first request triggers the same API-key
prompt in the terminal running the server. Use the **Reset** button to clear
conversation memory.

To skip the prompt, export the key first: `export GOOGLE_API_KEY="your-key-here"`.

### Inspecting the chunks

```bash
uv run python ingest.py
```

Prints the chunk count and the first chunks, without calling Gemini.

## Expected outcome

The chatbot should answer these correctly, with the exact figure from the surrounding
paragraph and nothing that isn't in the report:

- How many depository libraries did GPO staff visit in FY 2018?
- What new digital system replaced FDsys, and when was it fully deployed?
- How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?

A question the report can't answer (e.g. about FY 2020) should get
"I could not find this in the document."
