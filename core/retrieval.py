import json
import re
import math
from functools import lru_cache
from pathlib import Path
import pymupdf as fitz
from core.llm import groq_chat, json_from_response

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "documents"
INDEX = ROOT / "data" / "index" / "pdf_index.json"


def _fingerprint(paths):
    return [(str(p.relative_to(DOCS)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]


def index_status():
    """Read cache metadata and PDF file stats only; never extracts page text."""
    DOCS.mkdir(parents=True, exist_ok=True)
    paths = sorted(DOCS.rglob("*.pdf"))
    files = _fingerprint(paths)
    try:
        cached = json.loads(INDEX.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    summary = cached.get("summary", {}) if cached else {}
    stale = bool(paths) and (not cached or cached.get("fingerprints") != files)
    if not paths:
        summary = {"documents": 0, "chunks": 0, "scanned": [], "errors": []}
        stale = False
    elif not summary:
        summary = {"documents": len(paths), "chunks": 0, "scanned": [], "errors": []}
    return {"summary": summary, "stale": stale, "indexed": bool(cached and not stale), "pdf_files": len(paths)}


def index_pdfs(force=False):
    DOCS.mkdir(parents=True, exist_ok=True)
    paths = sorted(DOCS.rglob("*.pdf"))
    fingerprints = _fingerprint(paths)
    if not force and INDEX.exists():
        try:
            cached = json.loads(INDEX.read_text(encoding="utf-8"))
            if cached.get("fingerprints") == fingerprints:
                return cached.get("summary", {"documents": 0, "chunks": 0, "scanned": [], "errors": []})
        except (OSError, json.JSONDecodeError):
            pass
    chunks, scanned, errors = [], [], []
    for path in paths:
        try:
            with fitz.open(path) as pdf:
                extracted = 0
                for page_no, page in enumerate(pdf, start=1):
                    text = re.sub(r"\s+", " ", page.get_text("text")).strip()
                    if not text: continue
                    extracted += len(text)
                    # Overlapping chunks preserve context while keeping retrieval precise.
                    for start in range(0, len(text), 850):
                        piece = text[start:start + 1050]
                        if piece:
                            chunks.append({"document": path.name, "relative": str(path.relative_to(DOCS)), "page": page_no, "language": "Hindi" if re.search(r"[\u0900-\u097F]", piece) else "English", "text": piece})
                if extracted == 0: scanned.append(path.name)
        except Exception as exc:
            errors.append(f"{path.name}: {exc.__class__.__name__}")
    summary = {"documents": len(paths), "chunks": len(chunks), "scanned": scanned, "errors": errors}
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps({"fingerprints": fingerprints, "chunks": chunks, "summary": summary}, ensure_ascii=False), encoding="utf-8")
    return summary


@lru_cache(maxsize=1)
def _cached_chunks(mtime_ns, size):
    try:
        return json.loads(INDEX.read_text(encoding="utf-8")).get("chunks", [])
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def _load_chunks():
    if not index_status()["indexed"]:
        return []
    try:
        stat = INDEX.stat()
        return _cached_chunks(stat.st_mtime_ns, stat.st_size)
    except OSError:
        return []


def _terms(text):
    return set(re.findall(r"[\w\u0900-\u097F]+", text.lower(), flags=re.UNICODE))


def _rank_chunks(query, chunks, language=None, limit=5):
    terms = _terms(query)
    if not terms: return []
    if not chunks: return []
    tokenized = [_terms(c.get("text", "")) for c in chunks]
    doc_freq = {term: sum(term in words for words in tokenized) for term in terms}
    avg_len = sum(len(words) for words in tokenized) / max(1, len(tokenized))
    numbers = {t for t in terms if any(ch.isdigit() for ch in t)}
    scored = []
    for chunk, words in zip(chunks, tokenized):
        overlap = terms & words
        if not overlap:
            continue
        length = max(1, len(words))
        score = 0.0
        for term in overlap:
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            frequency = sum(1 for word in re.findall(r"[\w\u0900-\u097F]+", chunk.get("text", "").casefold(), re.UNICODE) if word == term)
            score += idf * (frequency * 2.2) / (frequency + 1.2 * (0.25 + 0.75 * length / max(1, avg_len)))
        # Keep exact phrases and numeric facts visible in the ranking.
        if str(query).casefold().strip() in chunk.get("text", "").casefold():
            score += 1.0
        if numbers:
            score += 0.35 * len(numbers & words)
        if language and chunk.get("language") == language:
            score *= 1.08
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"title": item["document"], "document": item["document"], "location": f"page {item['page']}", "page": item["page"], "language": item["language"], "text": item["text"], "credibility": "primary document", "score": round(score, 3)} for score, item in scored[:limit]]


def retrieve_local(query, language=None, limit=5):
    return _rank_chunks(query, _load_chunks(), language, limit)


def generate_questions(api_key):
    chunks = _load_chunks()
    grouped = {}
    for chunk in chunks: grouped.setdefault(chunk["document"], []).append(chunk)
    out = []
    for document, doc_chunks in grouped.items():
        sample = "\n".join(f"[page {c['page']}] {c['text']}" for c in doc_chunks[:8])[:9000]
        try:
            data = json_from_response(groq_chat(api_key, "Create 2-3 concise questions answerable only from the supplied text. Return JSON {questions:[{question:string,page:number}]}.", sample))
            for q in data.get("questions", [])[:3]:
                out.append({"question": str(q["question"]), "page": int(q.get("page", doc_chunks[0]["page"])), "document": document})
        except Exception:
            continue
    return out, None if out else "No sample questions could be generated from the indexed text."
