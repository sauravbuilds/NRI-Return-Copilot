"""CA brief: built only from structured fields, never from raw user text."""
from guardrails.pii import check_pii


def build_brief(persona, answer) -> str:
    sit = ["Situation (no identifiers):"]
    if persona.relocation_status:
        sit.append(f"- Relocation status: {persona.relocation_status.replace('_', ' ')}")
    if persona.move_date:
        sit.append(f"- Move date: {persona.move_date:%B %Y}")
    if persona.countries_lived:
        sit.append(f"- Countries lived in: {', '.join(persona.countries_lived)}")
    if persona.days_in_india_by_year:
        sit.append("- Days in India: " + ", ".join(f"{d.year}: {d.days}" for d in persona.days_in_india_by_year))
    if persona.indian_accounts:
        sit.append(f"- Indian accounts: {', '.join(persona.indian_accounts)}")
    if persona.foreign_assets:
        sit.append(f"- Foreign assets: {', '.join(a.replace('_', ' ') for a in persona.foreign_assets)}")
    if persona.goals:
        sit.append(f"- Goals: {', '.join(g.replace('_', ' ') for g in persona.goals)}")
    if len(sit) == 1:
        sit.append("- No details were provided.")

    questions = [f"- {m.field}: {m.why_it_matters}" for m in answer.missing_information]
    questions += [f"- Review reason: {r}" for r in answer.review_reasons]
    if answer.abstained:
        questions.append("- The tool found no official text supporting an answer.")

    seen, pages = set(), []
    for item in [*answer.official_findings, *answer.roadmap]:
        for c in item.citations:
            if (c.doc_id, c.page) not in seen:
                seen.add((c.doc_id, c.page))
                pages.append(f"- {c.document_title} ({c.issuer}), page {c.page}: {c.page_url}")

    text = "\n".join(sit + ["", "Open questions:"] + (questions or ["- None listed."])
                     + ["", "Official pages worth reading:"] + (pages or ["- None retrieved."]))
    if check_pii(text).blocked:  # final safety net
        raise ValueError("CA brief failed the PII check")
    return text
