from core.llm import groq_chat, json_from_response


def extract_claims(answer, api_key, language="English"):
    system = "Extract atomic, independently verifiable factual claims. Split compound claims. Preserve the source language and all numbers, dates, units, conditions, exceptions, negation, and scope. Return only JSON: {\"claims\":[string]}"
    try:
        data = json_from_response(groq_chat(api_key, system, f"Language: {language}\nAnswer:\n{answer[:18000]}"))
        claims = data.get("claims", []) if isinstance(data, dict) else data
        if not isinstance(claims, list): raise ValueError("Claims response had an unexpected format.")
        return [str(c).strip() for c in claims if str(c).strip()], None
    except Exception as exc:
        return [], f"Claim extraction failed: {exc}"
