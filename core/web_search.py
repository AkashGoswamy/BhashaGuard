import requests
from html.parser import HTMLParser
import re
from urllib.parse import urlparse
from core.llm import record_api_call, record_api_failure

TAVILY_URL = "https://api.tavily.com/search"


class TextExtractor(HTMLParser):
    BLOCK_TAGS = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr", "td", "th", "section", "article"}
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"): self.skip += 1
        if tag in self.BLOCK_TAGS: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip: self.skip -= 1
        if tag in self.BLOCK_TAGS: self.parts.append("\n")
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
        page_ok = r.status_code == 200 and "html" in r.headers.get("content-type", "").lower()
        record_api_call("webpage", failed=not page_ok)
        if not page_ok: return ""
        parser = TextExtractor(); parser.feed(r.text)
        return " ".join(parser.parts)[:10000]
    except requests.RequestException:
        record_api_call("webpage", failed=True)
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
    if not ranked:
        return ""
    required_overlap = 1 if len(terms) < 3 else max(2, int(len(terms) * 0.1 + 0.999))
    ranked = [x for x in ranked if len(set(re.findall(r"[\w\u0900-\u097F]+", x[2].lower(), re.UNICODE)) & terms) >= required_overlap]
    if not ranked:
        return ""
    selected = sorted(sorted(ranked, key=lambda x: x[0], reverse=True)[:6], key=lambda x: x[1])
    passage = " ".join(x[2] for x in selected)
    query_numbers = {x for x in terms if any(ch.isdigit() for ch in x)}
    passage_terms = {x for x in re.findall(r"[\w\u0900-\u097F]+", passage.lower(), re.UNICODE)}
    if query_numbers and not (query_numbers & passage_terms):
        return ""
    return passage[:limit]


def search_web(claim, language="English", api_key=""):
    if not api_key: return [], "Tavily key not configured; live search unavailable."
    queries = [claim[:400]]
    # One search per original language; Hindi queries are kept intact.
    try:
        r = requests.post(TAVILY_URL, json={"api_key": api_key, "query": queries[0], "search_depth": "advanced", "max_results": 5, "include_answer": False}, timeout=25)
        record_api_call("tavily", failed=not r.ok)
        if r.status_code == 401: return [], "Tavily rejected the key; live search unavailable."
        if r.status_code == 429: return [], "Tavily rate limit reached; live search unavailable."
        r.raise_for_status()
        results = r.json().get("results", [])
    except requests.RequestException as exc:
        if 'r' not in locals(): record_api_call("tavily", failed=True)
        return [], f"Live search failed ({exc.__class__.__name__}); local PDF evidence may still be used."
    except (ValueError, TypeError, AttributeError):
        record_api_failure("tavily")
        return [], "Tavily returned an invalid response; local PDF evidence may still be used."
    evidence = []
    for item in results:
        url = item.get("url", "")
        domain = urlparse(url).hostname or ""
        host = domain.casefold().removeprefix("www.")
        credibility = "official web source" if (host.endswith(".gov") or host.endswith(".gov.in") or host.endswith(".nic.in") or host == "gov.in" or host == "nic.in") else "academic web source" if (host.endswith(".edu") or host.endswith(".ac.in")) else "web source"
        page = _page_text(url) if url else ""
        page_passage = _relevant_passage(page, claim) if page else ""
        snippet_passage = _relevant_passage(item.get("content", ""), claim) if not page_passage else ""
        body = page_passage or snippet_passage
        evidence.append({"title": item.get("title", "Web source"), "url": url, "domain": domain, "date": item.get("published_date", ""), "language": language, "text": body, "retrieved_page": bool(page_passage), "credibility": credibility if page_passage else "search result only" if snippet_passage else "irrelevant page"})
    note = "Webpage text retrieved when available; search-result-only passages are marked as such and are not treated as conclusive." if evidence else "Tavily returned no results."
    return evidence, note
