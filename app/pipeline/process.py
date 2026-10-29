"""The full submission pipeline, in order:

guard_input -> scrub PII -> ML classifier -> LLM analyse (rewrite + triage + safety flags) -> guard flags
-> re-scan rewrite -> route (rules) -> RAG explanation + reply -> store -> ticket thread
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app import db
from app.config import ESCALATION_ROUTES, ROUTE_LABELS, SENIOR_POOL
from app.pipeline.guardrails import check_input, check_llm_flags, pii_clean
from app.pipeline.classifier import predict as ml_predict
from app.pipeline.privacy import GENERIC, PLACEHOLDER, scrub
from app.pipeline.rag import Explanation, explain
from app.pipeline.triage import Analysis, analyse, route


@dataclass
class SubmitResult:
    accepted: bool
    message: str
    ticket_id: str | None = None
    final_text: str | None = None
    reply: str | None = None
    route: str | None = None
    trace: list | None = None          # step-by-step log for the demo "how it works" panel


def submit(text: str, department: str, team: str | None, level_band: str) -> SubmitResult:
    trace = []
    g = check_input(text)
    trace.append(("Input guardrails", "passed" if g.allowed else f"blocked: {', '.join(g.reasons)}"))
    if not g.allowed:
        db.log_blocked(g.reasons[0], "rules")
        return SubmitResult(False, g.message, trace=trace)

    t0 = time.time()
    s = scrub(text)
    trace.append(("PII scrubber", f"removed {len(s.spans)} item(s): {s.counts or 'none'} ({time.time() - t0:.1f}s)"))

    # Trained classifier (logistic regression): instant, free, runs on every note
    t0 = time.time()
    ml = ml_predict(s.text)
    if ml:
        trace.append(("ML classifier (logistic regression)",
                      f"{ml.category} ({ml.confidence:.0%}); P(needs committee) = {ml.escalate_prob:.2f} "
                      f"vs threshold {ml.threshold:.2f} -> {'escalate' if ml.escalate else 'no'} ({time.time() - t0:.2f}s)"))

    from app.llm import LAST, LLMError
    t0 = time.time()
    try:
        a = analyse(s.text)
        llm_ok = True
    except LLMError as e:
        if not ml:
            raise
        # AI service down: fall back to the trained model and rule-based anonymisation
        llm_ok = False
        a = Analysis(rewritten=PLACEHOLDER.sub(lambda m: GENERIC[m.group(1)], s.text), category=ml.category,
                     sentiment="negative", severity=4 if ml.escalate else 3,
                     summary=f"Employee raises a {ml.category.replace('_', ' ')} concern.")
        trace.append(("AI rewrite + triage", f"unavailable ({str(e)[:60]}); used ML classifier instead"))
    if llm_ok:
        fg = check_llm_flags(a.flags.model_dump())
        trace.append(("AI rewrite + triage", f"{a.category}, {a.sentiment}, severity {a.severity} "
                      f"({LAST['provider'] or 'mock'}, {time.time() - t0:.1f}s)"))
        if ml:
            trace.append(("LLM vs ML", "agree" if ml.category == a.category else
                          f"disagree (LLM {a.category}, ML {ml.category}); reviewer sees both"))
        if not fg.allowed:
            trace.append(("AI safety check", f"blocked: {', '.join(fg.reasons)}"))
            db.log_blocked(fg.reasons[0], "llm")
            return SubmitResult(False, fg.message, trace=trace)

    final, leaks = pii_clean(a.rewritten)
    trace.append(("Re-scan of rewrite", f"{leaks} leftover item(s) removed"))

    rt, reason = route(a.category, s.text)
    # Recall safety net: the trained model can escalate a case the LLM labelled as routine
    if rt == "hr" and ml and ml.escalate:
        rt = ESCALATION_ROUTES[ml.committee_category]
        reason = f"ML safety net: P(needs committee) {ml.escalate_prob:.2f} >= threshold {ml.threshold:.2f}"
    trace.append(("Routing (rules)", f"{ROUTE_LABELS[rt]} ({reason})"))

    t0 = time.time()
    try:
        exp, docs, ok = explain(final, a.category, rt)
        trace.append(("Policy retrieval (RAG)", f"retrieved §{', §'.join(d.id for d in docs)}; "
                      f"cited {', '.join('§' + c for c in exp.citations) or 'none'}; grounded={ok} "
                      f"({LAST['provider'] or 'mock'}, {time.time() - t0:.1f}s)"))
    except LLMError:
        exp, ok = Explanation(citations=[], explanation="Policy explanation unavailable (AI service busy).",
                              suggested_action="Review the concern and respond through the ticket.",
                              employee_reply=f"Thank you for raising this. It has been sent to the {ROUTE_LABELS[rt]}, "
                                             "who will respond through this ticket within 5 working days (§12.5)."), False
        trace.append(("Policy retrieval (RAG)", "AI service busy; standard acknowledgement sent"))

    # Broader grouping is applied at write time, so it can never be undone by a query
    if level_band == "senior_leadership":
        department, team = SENIOR_POOL, None
    elif level_band == "manager":
        team = None

    ticket = db.new_ticket_id()
    db.insert_submission({
        "ticket_id": ticket, "source": "live", "department": department, "team": team,
        "level_band": level_band, "week": db.this_week(), "text": final,
        "summary": pii_clean(a.summary)[0], "category": a.category, "sentiment": a.sentiment,
        "severity": a.severity, "route": rt, "route_reason": reason, "urgent": int(a.severity >= 5),
        "status": "submitted", "explanation": exp.explanation, "suggested_action": exp.suggested_action,
        "citations": exp.citations, "grounded": int(ok),
        "ml_category": ml.category if ml else None, "ml_confidence": ml.confidence if ml else None,
        "ml_escalate_prob": ml.escalate_prob if ml else None})
    db.add_message(ticket, "system", final, "Your feedback, as reviewers see it")
    db.add_message(ticket, "system", exp.employee_reply, "Automatic reply")
    db.log("system", "submission_routed", ticket, rt)
    return SubmitResult(True, "Submitted", ticket, final, exp.employee_reply, rt, trace)
