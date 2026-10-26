"""
PDF -> clean, sentence-aligned chunks for the GPO 2018 Annual Report.

Two fixes over a plain page.get_text() + fixed-size splitter:
  1. Reading order — two-column pages are read band by band: the left column top
     to bottom, then the right one. The running header/footer is dropped.
  2. Sentence chunks — whole sentences only, ~450 tokens each, with ~60 tokens of
     overlap, so a fact near a boundary appears whole in at least one chunk.

Run on its own to inspect the chunks:
    uv run python ingest.py
"""

import re

import pymupdf


def tokens(text):
    """Rough token count (~4 characters per token) — good enough for sizing chunks."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# 1. Reading order
# ---------------------------------------------------------------------------
def clean(text):
    text = re.sub(r"-\n(?=\w)", "-", text)       # "Vance-\nCooks" -> "Vance-Cooks"
    return re.sub(r"\s+", " ", text).strip()


def is_running_header(text, y0, page_height):
    """Footer band ("2 0 1 8 A N N U A L R E P O R T | ...") and bare page numbers."""
    return (y0 > page_height - 50
            or re.search(r"(\b\w ){5,}", text) is not None
            or re.fullmatch(r"\d{1,3}", text) is not None)


def reading_order(blocks):
    """Wide blocks (titles, banners) split the page into bands.
    Inside each band: the left column top to bottom, then the right column."""
    ordered, band = [], []

    def flush():
        left = sorted([b for b in band if b["left"]], key=lambda b: b["y0"])
        right = sorted([b for b in band if not b["left"]], key=lambda b: b["y0"])
        ordered.extend(left + right)
        band.clear()

    for b in sorted(blocks, key=lambda b: (b["y0"], b["x0"])):
        if b["wide"]:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered


def read_pages(path):
    """One entry per page: {"page": n, "text": text in reading order}."""
    pages = []
    for number, page in enumerate(pymupdf.open(path), start=1):
        width, height = page.rect.width, page.rect.height
        blocks = []
        for x0, y0, x1, y1, text, _, kind in page.get_text("blocks"):
            text = clean(text)
            if kind != 0 or not text or is_running_header(text, y0, height):
                continue
            blocks.append({"x0": x0, "y0": y0, "text": text,
                           "wide": (x1 - x0) > width * 0.55, "left": x0 < width / 2})
        text = " ".join(b["text"] for b in reading_order(blocks))
        if text:
            pages.append({"page": number, "text": text})
    return pages


# ---------------------------------------------------------------------------
# 2. Sentence chunks
# ---------------------------------------------------------------------------
ABBREVIATION = re.compile(r"\b(?:U\.S|Mr|Ms|Mrs|Dr|No|St|Inc|Jr|vs|e\.g|i\.e|Jan|Feb|Aug|Sept|Oct|Nov|Dec)\.$")


def split_sentences(text):
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])|(?<=[.!?][”\")])\s+(?=[A-Z0-9“\"(])", text)
    sentences = []
    for piece in pieces:
        if sentences and ABBREVIATION.search(sentences[-1]):
            sentences[-1] += " " + piece                # "U.S. Government" is one sentence
        else:
            sentences.append(piece)
    return [s for s in sentences if s.strip()]


def sentence_chunks(pages, target=450, overlap=60):
    """Whole sentences only, about `target` tokens each. The last ~`overlap` tokens
    repeat at the start of the next chunk, so a fact near a boundary appears whole somewhere."""
    sents = []
    for p in pages:
        for s in split_sentences(p["text"]):
            words = s.split()
            while tokens(s) > target:                   # a run-on "sentence" (lists, tables)
                half = len(words) // 2
                sents.append((" ".join(words[:half]), p["page"]))
                words = words[half:]
                s = " ".join(words)
            sents.append((s, p["page"]))

    chunks, current, size = [], [], 0

    def close():
        nonlocal current, size
        chunks.append({"id": len(chunks), "text": " ".join(s for s, _ in current),
                       "page_start": current[0][1], "page_end": current[-1][1]})
        keep, kept = [], 0
        for s, pg in reversed(current):                 # carry the tail over as overlap
            if kept + tokens(s) > overlap:
                break
            keep.insert(0, (s, pg))
            kept += tokens(s)
        current, size = keep, kept

    for s, pg in sents:
        if size + tokens(s) > target and size > overlap:
            close()
        current.append((s, pg))
        size += tokens(s)
    if size > overlap or not chunks:
        close()
    return chunks


def build_chunks(path):
    return sentence_chunks(read_pages(path))


if __name__ == "__main__":
    import os

    pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GPO-Anual-report.pdf")
    chunks = build_chunks(pdf)
    ends = sum(c["text"].rstrip("”\")").endswith((".", "!", "?")) for c in chunks)
    print(f"{len(chunks)} chunks, {ends} end on a full sentence\n")
    for c in chunks[:3]:
        print(f"--- chunk {c['id']} (pages {c['page_start']}-{c['page_end']}, ~{tokens(c['text'])} tokens)")
        print(c["text"], "\n")
