import json
import re
from pathlib import Path
import pymupdf as fitz
from core.llm import groq_chat, json_from_response

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "documents"
INDEX = ROOT / "data" / "index" / "pdf_index.json"


def _fingerprint(paths):
    return [(str(p.relative_to(DOCS)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]


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


def _load_chunks():
    index_pdfs()
    try: return json.loads(INDEX.read_text(encoding="utf-8")).get("chunks", [])
    except (OSError, json.JSONDecodeError): return []


def _terms(text):
    return set(re.findall(r"[\w\u0900-\u097F]+", text.lower(), flags=re.UNICODE))


def retrieve_local(query, language=None, limit=5):
    terms = _terms(query)
    if not terms: return []
    scored = []
    for chunk in _load_chunks():
        cterms = _terms(chunk["text"])
        overlap = len(terms & cterms)
        if overlap:
            # Jaccard-like ranking limits long chunks from dominating.
            score = overlap / (len(terms) ** 0.5 * max(1, len(cterms)) ** 0.25)
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"title": item["document"], "document": item["document"], "location": f"page {item['page']}", "page": item["page"], "language": item["language"], "text": item["text"], "credibility": "primary document", "score": round(score, 3)} for score, item in scored[:limit]]


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
