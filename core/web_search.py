import requests
from html.parser import HTMLParser
import re

TAVILY_URL = "https://api.tavily.com/search"


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"): self.skip += 1
    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip: self.skip -= 1
    def handle_data(self, data):
        if not self.skip and data.strip(): self.parts.append(data.strip())


def validate_tavily_key(api_key):
    try:
        r = requests.post(TAVILY_URL, json={"api_key": api_key, "query": "BhashaGuard API validation", "max_results": 1}, timeout=15)
        if r.status_code == 200: return True, "Key validated."
        if r.status_code in (401, 403): return False, "Invalid API key."
        return False, f"Service returned HTTP {r.status_code}."
    except requests.RequestException as exc:
        return False, f"Could not connect: {exc.__class__.__name__}."


def _page_text(url):
    try:
        r = requests.get(url, headers={"User-Agent": "BhashaGuard/1.0 (evidence auditor)"}, timeout=12)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "").lower(): return ""
        parser = TextExtractor(); parser.feed(r.text)
        return " ".join(parser.parts)[:10000]
    except requests.RequestException:
        return ""


def _relevant_passage(text, query, limit=2200):
    """Return a short passage from the retrieved page relevant to the query."""
    terms = {x for x in re.findall(r"[\w\u0900-\u097F]+", query.lower(), re.UNICODE) if len(x) > 2}
    pieces = [p.strip() for p in re.split(r"\n+|(?<=[.!?।])\s+", text) if p.strip()]
    ranked = []
    for order, piece in enumerate(pieces):
        words = set(re.findall(r"[\w\u0900-\u097F]+", piece.lower(), re.UNICODE))
        overlap = len(words & terms)
        if overlap:
            ranked.append((overlap / max(1, len(terms)), order, piece))
    selected = sorted(sorted(ranked, key=lambda x: x[0], reverse=True)[:6], key=lambda x: x[1])
    passage = " ".join(x[2] for x in selected)
    if not passage:
        passage = text[:800]
    return passage[:limit]


def search_web(claim, language="English", api_key=""):
    if not api_key: return [], "Tavily key not configured; live search unavailable."
    queries = [claim[:400]]
    # One search per original language; Hindi queries are kept intact.
    try:
        r = requests.post(TAVILY_URL, json={"api_key": api_key, "query": queries[0], "search_depth": "advanced", "max_results": 5, "include_answer": False}, timeout=25)
        if r.status_code == 401: return [], "Tavily rejected the key; live search unavailable."
        if r.status_code == 429: return [], "Tavily rate limit reached; live search unavailable."
        r.raise_for_status()
        results = r.json().get("results", [])
    except requests.RequestException as exc:
        return [], f"Live search failed ({exc.__class__.__name__}); local PDF evidence may still be used."
    evidence = []
    for item in results:
        url = item.get("url", "")
        page = _page_text(url) if url else ""
        body = _relevant_passage(page, claim) if page else _relevant_passage(item.get("content", ""), claim)
        evidence.append({"title": item.get("title", "Web source"), "url": url, "domain": url.split("/")[2] if "://" in url else "", "date": item.get("published_date", ""), "language": language, "text": body, "retrieved_page": bool(page), "credibility": "web source" if page else "search result only"})
    note = "Webpage text retrieved when available; search-result-only passages are marked as such and are not treated as conclusive." if evidence else "Tavily returned no results."
    return evidence, note
