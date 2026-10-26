"""
GPO 2018 Annual Report Q&A — Use Case 1: plain-text business report (the baseline case)

  Question -> Hybrid search (dense + BM25, fused with RRF) -> Gemini rerank
           -> Numbered passages + Conversation History -> Gemini -> Cited answer

Ingestion (reading order + sentence chunks) lives in ingest.py. The chunks and their
vectors are cached in .index/ after the first run, so later starts skip re-embedding.

Run:
    uv run python chatbot.py
"""

import os
import re
import json
import logging
from getpass import getpass

# Silence the google-genai SDK's benign "use Chat.send_message instead" advisory
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding

from ingest import build_chunks


# ---------------------------------------------------------------------------
# 1. API key (never hardcoded)
# ---------------------------------------------------------------------------
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("Enter your Gemini API key: ")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def get_text(response):
    """Return plain text of an LLM response, whether `.content` is a string
    or a list of content blocks (e.g. text + an internal thought-signature block)."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# ---------------------------------------------------------------------------
# 2. LLM — temperature 0: answers must quote the document, not paraphrase it
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


# ---------------------------------------------------------------------------
# 3. Embeddings + vector store (built once, then loaded from .index/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "GPO-Anual-report.pdf")
INDEX_DIR = os.path.join(BASE_DIR, ".index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")
VECTORS_PATH = os.path.join(INDEX_DIR, "vectors.json")


class LocalEmbeddings(Embeddings):
    """BAAI/bge-small-en-v1.5 via fastembed: runs locally on CPU (ONNX), so indexing has
    no API quota. The model (~67 MB) is downloaded once on first use."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.query_embed(text))).tolist()


embeddings = LocalEmbeddings()

if os.path.exists(CHUNKS_PATH) and os.path.exists(VECTORS_PATH):
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    vector_store = InMemoryVectorStore.load(VECTORS_PATH, embeddings)
else:
    chunks = build_chunks(PDF_PATH)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(
        [Document(page_content=c["text"], metadata={"id": c["id"]}) for c in chunks],
        ids=[str(c["id"]) for c in chunks],
    )
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    vector_store.dump(VECTORS_PATH)


# ---------------------------------------------------------------------------
# 4. Hybrid search: meaning (dense) + exact words (BM25), merged by rank
# ---------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

RRF_K = 60  # Reciprocal Rank Fusion constant from the original paper


def dense_search(question, n):
    return [doc.metadata["id"] for doc in vector_store.similarity_search(question, k=n)]


def bm25_search(question, n):
    scores = bm25.get_scores(tokenize(question))
    return sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:n]


def hybrid_search(question, k=20):
    """Each chunk scores 1/(RRF_K + rank) in every list it appears in; the scores are
    added. Only ranks count, so cosine similarity and BM25 never need a common scale."""
    fused = {}
    for ranking in (dense_search(question, k), bm25_search(question, k)):
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (RRF_K + rank)
    return [chunks[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]


# ---------------------------------------------------------------------------
# 5. Rerank: Gemini reads the question and the whole shortlist together
# ---------------------------------------------------------------------------
RERANK_PROMPT = (
    "You rank passages by how directly they answer a question.\n"
    "Reply with JSON only: a list of passage numbers, best first, like [3, 0, 7]. "
    "List at most {k} passages."
)


def rerank(question, hits, k=5):
    numbered = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits))
    reply = get_text(llm.invoke([
        SystemMessage(content=RERANK_PROMPT.format(k=k)),
        HumanMessage(content=f"Question: {question}\n\nPassages:\n\n{numbered}"),
    ]))
    found = re.search(r"\[[\d,\s]*\]", reply)
    order = [i for i in json.loads(found.group(0)) if 0 <= i < len(hits)] if found else []
    order = list(dict.fromkeys(order))                       # drop repeats, keep order
    order += [i for i in range(len(hits)) if i not in order]  # fall back to fused order
    return [hits[i] for i in order[:k]]


# ---------------------------------------------------------------------------
# 6. Grounded RAG + memory chatbot
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You answer questions about the U.S. Government Publishing Office (GPO) 2018 Annual Report "
    "using ONLY the numbered context passages.\n"
    "- Use only facts stated in the passages, or already established earlier in this conversation. "
    "Never add outside knowledge.\n"
    "- Quote numbers, names and dates exactly as written.\n"
    "- Cite the passages you used with square brackets only, like [1] or [2][3]. Plain text, no bold.\n"
    "- If passages give different figures, report each one with its context.\n"
    "- If the passages do not contain the answer, reply exactly: I could not find this in the document.\n"
    "- Be brief: one to three sentences."
)


def build_context(hits):
    """Number the passages and show where each came from, so the answer can cite them."""
    return "\n\n".join(
        f"[{i}] (pages {h['page_start']}-{h['page_end']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


conversation_history = []  # list of HumanMessage / AIMessage, growing each turn


def rag_chat(user_message):
    # 1. Retrieve: hybrid shortlist of 20, reranked down to the best 5
    hits = rerank(user_message, hybrid_search(user_message))

    # 2. Build fresh instructions (including the numbered passages) for this turn
    system_message = SystemMessage(content=SYSTEM_PROMPT + "\n\nCONTEXT PASSAGES:\n\n" + build_context(hits))

    # 3. Combine: system instructions + everything said so far + the new message
    messages = [system_message] + conversation_history + [HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    reply_text = get_text(response)

    # 4. Record this turn so future turns remember it
    conversation_history.append(HumanMessage(content=user_message))
    conversation_history.append(AIMessage(content=reply_text))

    return reply_text


# ---------------------------------------------------------------------------
# 7. Interactive loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("GPO 2018 Annual Report assistant ready. Type 'quit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "quit":
            break
        print("Assistant:", rag_chat(user_input))
