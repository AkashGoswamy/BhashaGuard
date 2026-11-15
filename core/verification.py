from core.llm import groq_chat, json_from_response

VERDICTS = {"ACCEPTED", "BLOCKED", "NEEDS REVIEW"}


def verify_claim(claim, evidence, api_key):
    if not evidence:
        return {"verdict": "NEEDS REVIEW", "strength": "INSUFFICIENT", "explanation": "No relevant evidence was retrieved; lack of evidence does not establish that the claim is false.", "correction": ""}
    sources = []
    for i, item in enumerate(evidence, start=1):
        sources.append(f"SOURCE {i} ({item.get('credibility','unknown')}; {item.get('title','')}; {item.get('location',item.get('url',''))}; {item.get('date','')}): {item.get('text','')[:2200]}")
    system = ("You assess claims only from supplied evidence. Search snippets alone are weak and cannot conclusively prove a claim. "
              "Primary official/academic sources are stronger. Distinguish unrelated contexts, dates, units, conditions and exceptions. "
              "No evidence or ambiguous/conflicting evidence means NEEDS REVIEW. Return only JSON with verdict (ACCEPTED, BLOCKED, NEEDS REVIEW), "
              "strength (STRONG, MODERATE, INSUFFICIENT), explanation, correction (string or empty). Never use confidence as evidence. "
              "ACCEPTED requires sufficient direct supporting evidence; BLOCKED requires sufficient direct contradictory evidence. Suggest correction only if directly supported.")
    try:
        data = json_from_response(groq_chat(api_key, system, f"CLAIM:\n{claim}\n\nEVIDENCE:\n" + "\n\n".join(sources)))
        verdict = str(data.get("verdict", "NEEDS REVIEW")).upper()
        strength = str(data.get("strength", "INSUFFICIENT")).upper()
        if verdict not in VERDICTS: verdict = "NEEDS REVIEW"
        if strength not in {"STRONG", "MODERATE", "INSUFFICIENT"}: strength = "INSUFFICIENT"
        if verdict == "NEEDS REVIEW": data["correction"] = ""
        # A snippet-only result cannot establish a strong finding.
        if all(x.get("credibility") == "search result only" for x in evidence) and verdict != "NEEDS REVIEW":
            verdict, strength = "NEEDS REVIEW", "INSUFFICIENT"
            data["explanation"] = "Only search-result snippets were retrievable; they are insufficient to establish this verdict. " + str(data.get("explanation", ""))
            data["correction"] = ""
        return {"verdict": verdict, "strength": strength, "explanation": str(data.get("explanation", "Evidence assessment unavailable.")), "correction": str(data.get("correction", ""))}
    except Exception as exc:
        return {"verdict": "NEEDS REVIEW", "strength": "INSUFFICIENT", "explanation": f"Evidence assessment failed ({exc}).", "correction": ""}
