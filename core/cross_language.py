from core.llm import groq_chat, json_from_response
from core.extraction import extract_claims


def compare_answers(english, hindi, api_key, mode="English and Hindi answers"):
    system = ("Compare the meaning of the two answers, preserving exact quantities, dates, conditions, exceptions, negation and scope. "
              "List material and minor factual discrepancies, missing conditions, omissions, and claims unique to one side. "
              "Do not decide factual truth from agreement. Return JSON {discrepancies:[{type,importance,english,hindi,explanation}], "
              "english_claims:[string],hindi_claims:[string]}. Importance is MATERIAL or MINOR.")
    try:
        data = json_from_response(groq_chat(api_key, system, f"Input type: {mode}\nENGLISH:\n{english[:12000]}\n\nHINDI:\n{hindi[:12000]}"))
        if not isinstance(data, dict) or not isinstance(data.get("discrepancies", []), list): raise ValueError("Unexpected comparison format.")
        # Extract each language independently. Comparison models sometimes
        # omit one list, especially for short Hindi answers; never silently
        # skip the independent evidence check for that side.
        en_claims, en_error = extract_claims(english, api_key, "English")
        hi_claims, hi_error = extract_claims(hindi, api_key, "Hindi")
        data["english_claims"] = en_claims
        data["hindi_claims"] = hi_claims
        data["claim_extraction_errors"] = [x for x in (en_error, hi_error) if x]
        for d in data["discrepancies"]:
            if d.get("importance") not in {"MATERIAL", "MINOR"}: d["importance"] = "MATERIAL"
        return data, None
    except Exception as exc:
        return {}, f"Comparison failed: {exc}"
