"""AI-Powered Anonymous Employee Feedback System: Streamlit app.

Run:  streamlit run app/streamlit_app.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app import db, insights as I
from app.config import (CATEGORY_LABELS, DIVISIONS, K_ANONYMITY_MIN as K, ROUTE_LABELS,
                        SENIOR_POOL)
from app.llm import active_provider

st.set_page_config(page_title="Anonymous Feedback", page_icon="🔒", layout="wide")
db.init()


@st.cache_resource
def _auto_seed():
    """On a fresh deployment (e.g. Streamlit Cloud), load the demo data once."""
    from app.config import DATA_DIR
    if not db.query("SELECT 1 FROM submissions LIMIT 1") and (DATA_DIR / "synthetic" / "feedback_raw.jsonl").exists():
        from scripts.seed_demo_db import main as seed
        seed()
    return True


@st.cache_resource(show_spinner="Loading privacy and policy models (first start only)...")
def _warm_up():
    """Load Presidio/spaCy and build the handbook index once at startup, not on the first submission."""
    from app.pipeline.privacy import scrub
    from app.pipeline.retrieval import handbook_index
    scrub("Warm-up text for Rahul in Pune.")
    return handbook_index().backend


_auto_seed()
_warm_up()

ROLES = {
    "Employee": "employee",
    "HR Business Partner": "hr",
    "POSH Internal Committee": "posh_committee",
    "Ethics Committee": "ethics_committee",
    "Senior Review Panel": "senior_review_panel",
    "Team Manager": "manager",
    "Department Head": "dept_head",
    "Leadership (CEO / CXO)": "leadership",
    "Admin": "admin",
}
COMPANY = {
    "Engineering": ["Payments", "Platform", "Mobile", "Data", "QA"],
    "Sales": ["Enterprise Sales", "Inside Sales", "Channel Partners"],
    "Operations": ["Supply Chain", "Facilities", "Logistics"],
    "Customer Support": ["Tier 1 Support", "Escalations"],
    "Marketing": ["Brand", "Growth Marketing"],
    "Finance": ["Accounts", "Procurement"],
    "Product": ["Product Management", "Design"],
    "Human Resources": ["HR Business Partners", "Talent Acquisition"],
}
SEV_LABEL = {1: "1 · minor", 2: "2 · low", 3: "3 · moderate", 4: "4 · high", 5: "5 · critical"}

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.markdown("### 🔒 Anonymous Feedback")
    role_name = st.radio("View as", list(ROLES), index=0)
    role = ROLES[role_name]
    st.caption("Demo role switcher. In production, reviewers sign in with company SSO; "
               "employees never sign in.")
    st.divider()
    st.caption(f"AI provider: **{active_provider()}**  \nk-anonymity threshold: **{K}**")

# ------------------------------------------------------------------ helpers


def chart_categories(cat_df: pd.DataFrame, title: str):
    fig = px.bar(cat_df, x="n", y="Category", orientation="h", title=title,
                 labels={"n": "Responses", "Category": ""})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def status_badge(s: str) -> str:
    return {"submitted": "🟡", "acknowledged": "🔵", "in_review": "🟣", "action_taken": "🟢", "closed": "⚪"}.get(s, "") \
        + " " + db.STATUS_LABELS.get(s, s)


def thread(ticket: str, viewer: str):
    for m in db.messages(ticket):
        who = {"system": "🤖", "reviewer": "🧑‍💼", "employee": "🙂"}.get(m["sender"], "")
        label = m["label"] or m["sender"].title()
        with st.chat_message("assistant" if m["sender"] != "employee" else "user", avatar=who):
            st.caption(label)
            st.write(m["body"])


def log_once(role_key: str, action: str, ticket: str):
    k = f"logged:{role_key}:{action}:{ticket}"
    if k not in st.session_state:
        db.log(role_key, action, ticket)
        st.session_state[k] = True


def empty_db_notice():
    if not db.query("SELECT 1 FROM submissions LIMIT 1"):
        st.info("No feedback yet. For demo data run:  `python -m scripts.seed_demo_db`")
        return True
    return False


# ================================================================== EMPLOYEE

def employee_view():
    st.title("Share feedback anonymously")
    t1, t2, t3 = st.tabs(["✍️ Give feedback", "🎫 Check my ticket", "🛡️ How you're protected"])

    with t1:
        st.write("No login. Your name, email, device and IP address are never recorded. "
                 "You'll get a ticket ID to read replies, and only you will have it.")
        c1, c2, c3 = st.columns(3)
        dept = c1.selectbox("Department", list(COMPANY))
        level = c3.radio("Your level", ["Individual contributor", "Manager", "Senior leader"],
                         help="Managers are never grouped by team; senior leaders are pooled company-wide.")
        band = {"Individual contributor": "ic", "Manager": "manager", "Senior leader": "senior_leadership"}[level]
        team = c2.selectbox("Team (optional)", ["Prefer not to say"] + COMPANY[dept],
                            disabled=band != "ic")
        text = st.text_area("What would you like to raise?", height=170, max_chars=3000,
                            placeholder="Describe the issue, how often it happens and how it affects you. "
                                        "You can write in English or Hinglish.")
        show_trace = st.checkbox("Show how the system processed this (demo)", value=True)

        if st.button("Submit anonymously", type="primary"):
            recent = [t for t in st.session_state.get("subs", []) if time.time() - t < 600]
            if len(recent) >= 3:
                st.warning("You've submitted 3 notes in 10 minutes. Please wait a little before sending more.")
                return
            from app.pipeline.process import submit
            with st.spinner("Removing identifying details, rewriting and routing your feedback..."):
                try:
                    res = submit(text, dept, None if team == "Prefer not to say" else team, band)
                except Exception as e:
                    st.error("The AI service is busy (free-tier limit). Please try again in a minute.")
                    st.caption(str(e)[:200])
                    return
            if not res.accepted:
                st.error(res.message)
            else:
                st.session_state["subs"] = recent + [time.time()]
                st.success("Submitted. Save your ticket ID: it is the only way to see replies.")
                st.code(res.ticket_id, language=None)
                st.markdown("**What reviewers will see** (your original wording is not stored):")
                st.info(res.final_text)
                st.markdown(f"**Routed to:** {ROUTE_LABELS[res.route]}")
                st.markdown("**Automatic reply**")
                st.write(res.reply)
            if show_trace and res.trace:
                with st.expander("Pipeline trace", expanded=not res.accepted):
                    for step, detail in res.trace:
                        st.markdown(f"- **{step}:** {detail}")

    with t2:
        tid = st.text_input("Ticket ID", placeholder="TKT-XXXXXX").strip().upper()
        if tid:
            s = db.get_submission(tid)
            if not s:
                st.error("No ticket with that ID.")
                return
            st.markdown(f"**Status:** {status_badge(s['status'])}  ·  **Handled by:** {ROUTE_LABELS[s['route']]}")
            thread(tid, "employee")
            msg = st.text_area("Reply to the reviewer", key="emp_reply")
            if st.button("Send reply") and msg.strip():
                from app.pipeline.guardrails import check_input, pii_clean
                g = check_input(msg)
                if not g.allowed and g.reasons[0] != "too_short":
                    st.error(g.message)
                else:
                    db.add_message(tid, "employee", pii_clean(msg)[0], "You (anonymous)")
                    st.rerun()
            if s["status"] in ("action_taken", "closed"):
                st.markdown("**Are you satisfied with how this was handled?**")
                a, b = st.columns(2)
                if a.button("👍 Satisfied"):
                    db.update_submission(tid, rating="satisfied", status="closed")
                    db.log("employee", "rated_satisfied", tid)
                    st.rerun()
                if b.button("👎 Not satisfied, escalate"):
                    db.update_submission(tid, rating="not_satisfied", status="in_review")
                    db.add_message(tid, "system", "Escalated one level for further review (§9.4).", "System")
                    db.log("employee", "escalated", tid)
                    st.rerun()

    with t3:
        st.markdown(f"""
1. **No identity is collected.** No login, name, email, IP address or device ID. Time is stored only as the calendar week.
2. **Identifying details are removed.** Names, employee IDs, emails, phone numbers, team, city and project names are scrubbed.
3. **Your writing style is hidden.** AI rewrites your note in neutral language before anyone reads it; your original text is discarded.
4. **Small groups are never shown.** Managers and leaders only see summaries for groups with at least {K} responses. Managers are not grouped by team, and senior leaders are pooled company-wide.
5. **Sensitive cases skip your management chain.** Harassment goes to the POSH Internal Committee, safety and fraud to the Ethics Committee, and complaints about HR or leadership to an independent Senior Review Panel.
6. **Retaliation is prohibited** (Handbook §12.6), and trying to identify an author is serious misconduct.
""")


# ================================================================== REVIEWERS

def reviewer_view(route_key: str):
    title = ROUTE_LABELS[route_key]
    st.title(f"{title}: case queue")
    if empty_db_notice():
        return
    tabs = ["📥 Queue", "📄 Case"] + (["📊 Insights", "💬 Ask the feedback"] if route_key == "hr" else [])
    tab = st.tabs(tabs)
    df = I.load()
    q = df[df["route"] == route_key].copy()
    cw = db.this_week()
    q["overdue"] = (q["status"] == "submitted") & q["week"].map(lambda w: db.weeks_between(w, cw) >= 1)

    with tab[0]:
        c1, c2, c3, c4 = st.columns(4)
        sts = c1.multiselect("Status", db.STATUSES, default=["submitted", "acknowledged", "in_review"],
                             format_func=lambda s: db.STATUS_LABELS[s])
        cats = c2.multiselect("Category", sorted(q["category"].unique()), format_func=CATEGORY_LABELS.get)
        depts = c3.multiselect("Department", sorted(q["department"].unique()))
        min_sev = c4.slider("Min severity", 1, 5, 1)
        v = q[q["status"].isin(sts) & (q["severity"] >= min_sev)]
        if cats:
            v = v[v["category"].isin(cats)]
        if depts:
            v = v[v["department"].isin(depts)]
        v = v.sort_values(["urgent", "overdue", "severity", "week"], ascending=[False, False, False, True])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Open cases", int(q["status"].isin(["submitted", "acknowledged", "in_review"]).sum()))
        m2.metric("Urgent (severity 5)", int((q["urgent"] == 1).sum()))
        m3.metric("Overdue acknowledgement", int(q["overdue"].sum()))
        m4.metric("Showing", len(v))
        show = v[["ticket_id", "week", "department", "category", "severity", "status", "urgent", "overdue", "text"]].copy()
        show["category"] = show["category"].map(CATEGORY_LABELS)
        show["status"] = show["status"].map(db.STATUS_LABELS)
        show["text"] = show["text"].str.slice(0, 110) + "…"
        st.dataframe(show, use_container_width=True, hide_index=True, height=420,
                     column_config={"urgent": st.column_config.CheckboxColumn("Urgent"),
                                    "overdue": st.column_config.CheckboxColumn("Overdue")})
        st.session_state["queue_ids"] = list(v["ticket_id"])

    with tab[1]:
        ids = st.session_state.get("queue_ids") or list(q["ticket_id"])
        if not ids:
            st.write("No cases match the queue filters.")
        else:
            tid = st.selectbox("Open case", ids)
            case_view(tid, route_key, df)

    if route_key == "hr":
        with tab[2]:
            insights_panel(I.line_view(df))   # committee cases stay with the committee
        with tab[3]:
            st.write("Ask about themes across anonymised feedback. Answers never quote or identify individuals.")
            question = st.text_input("Question", placeholder="What are the top concerns in Sales this quarter?")
            if st.button("Ask", type="primary") and question:
                with st.spinner("Reading the relevant feedback..."):
                    try:
                        ans, n, scope = I.ask(question, I.line_view(df))
                        st.markdown(ans)
                        if n:
                            st.caption(f"Based on {n} anonymised notes · scope: {scope}")
                        db.log("hr", "ask", "", question[:120])
                    except Exception as e:
                        st.error("The AI service is busy. Try again in a minute.")
                        st.caption(str(e)[:200])


def case_view(tid: str, route_key: str, df: pd.DataFrame):
    s = db.get_submission(tid)
    log_once(route_key, "viewed_case", tid)
    left, right = st.columns([3, 2])
    with left:
        st.subheader(tid)
        if s["source"] == "seed":
            st.caption("Demo case (seeded with rule-based anonymisation; live submissions are AI-rewritten).")
        st.info(s["text"])
        a, b, c = st.columns(3)
        a.metric("Category", CATEGORY_LABELS[s["category"]])
        b.metric("Severity", SEV_LABEL[s["severity"]])
        c.metric("Sentiment", s["sentiment"].title())
        st.caption(f"Week {s['week']} · {s['department']}{' · ' + s['team'] if s['team'] else ''} · "
                   f"routing: {s['route_reason']}")

        st.markdown("#### Policy explanation")
        if not s.get("explanation"):
            if st.button("Generate cited explanation (AI)"):
                from app.pipeline.rag import explain
                with st.spinner("Retrieving policy clauses..."):
                    try:
                        exp, docs, ok = explain(s["text"], s["category"], s["route"])
                        db.update_submission(tid, explanation=exp.explanation, suggested_action=exp.suggested_action,
                                             citations=exp.citations, grounded=int(ok))
                        st.rerun()
                    except Exception as e:
                        st.error("AI service busy; try again shortly.")
                        st.caption(str(e)[:200])
        else:
            st.write(s["explanation"])
            st.markdown(f"**Suggested action:** {s['suggested_action']}")
            if s["citations"]:
                from app.pipeline.retrieval import handbook_chunks
                chunks = {d.id: d for d in handbook_chunks()}
                with st.expander("Cited clauses: " + ", ".join("§" + c for c in s["citations"])):
                    for c in s["citations"]:
                        if c in chunks:
                            st.markdown(f"**§{c} {chunks[c].meta['title']}**  \n{chunks[c].text.split(chr(10), 1)[1]}")
            elif s.get("grounded") == 0:
                st.caption("No policy clause applies; the AI did not invent one.")

        recent = I.recent_weeks(df, 8)
        similar = recent[(recent["department"] == s["department"]) & (recent["category"] == s["category"])]
        if len(similar) >= 3:
            st.warning(f"**Pattern:** {len(similar)} similar {CATEGORY_LABELS[s['category']].lower()} cases from "
                       f"{s['department']} in the last 8 weeks. Consider acting on the pattern rather than "
                       f"this case alone, which also protects the writer.")

    with right:
        st.markdown("#### Conversation")
        thread(tid, route_key)
        reply = st.text_area("Reply to employee (anonymous thread)", key=f"r_{tid}")
        new_status = st.selectbox("Status", db.STATUSES, index=db.STATUSES.index(s["status"]),
                                  format_func=lambda x: db.STATUS_LABELS[x], key=f"s_{tid}")
        if st.button("Send / update", type="primary"):
            if reply.strip():
                from app.pipeline.guardrails import pii_clean
                db.add_message(tid, "reviewer", pii_clean(reply)[0], ROUTE_LABELS[route_key])
                db.log(route_key, "replied", tid)
            if new_status != s["status"]:
                db.update_submission(tid, status=new_status)
                db.log(route_key, f"status:{new_status}", tid)
            st.rerun()
        with st.expander("Re-route (fix a misclassification)"):
            other = [r for r in ROUTE_LABELS if r != s["route"]]
            dest = st.selectbox("Move to", other, format_func=ROUTE_LABELS.get, key=f"rr_{tid}")
            if st.button("Re-route case"):
                db.update_submission(tid, route=dest, route_reason=f"re-routed by {ROUTE_LABELS[route_key]}")
                db.log(route_key, f"rerouted:{dest}", tid)
                st.success("Case moved. It no longer appears in this queue.")


# ================================================================== INSIGHTS (HR)

def insights_panel(df: pd.DataFrame):
    st.caption(f"Cells and groups with fewer than {K} responses are hidden.")
    c1, c2 = st.columns(2)
    with c1:
        hm = I.heatmap(df)
        fig = px.imshow(hm, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
                        title="Responses by department and category (last 13 weeks)")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Rising and falling issues** (last 8 weeks vs previous 8)")
        st.dataframe(I.rising_issues(df).rename(columns={"department": "Department"}), hide_index=True,
                     use_container_width=True, height=180)
        st.markdown("**Patterns HR can act on** (same department and category, last 8 weeks)")
        p = I.patterns(df)[["department", "Category", "cases"]].rename(
            columns={"department": "Department", "cases": "Cases"})
        st.dataframe(p.head(8), hide_index=True, use_container_width=True, height=180)
    t = df.groupby(["month", "category"]).size().reset_index(name="n")
    t["Category"] = t["category"].map(CATEGORY_LABELS)
    fig = px.line(t, x="month", y="n", color="Category", markers=True, title="Responses per month by category",
                  labels={"n": "Responses", "month": ""})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ================================================================== MANAGER / HEAD / LEADERSHIP

def manager_view():
    st.title("Team summary")
    st.caption(f"Individual notes are never shown. A summary appears only when at least {K} people have responded "
               "in the last 13 weeks. Counts are rounded. " + I.COMMITTEE_NOTE)
    if empty_db_notice():
        return
    c1, c2 = st.columns(2)
    dept = c1.selectbox("Department", list(COMPANY))
    team = c2.selectbox("Your team", COMPANY[dept])
    df = I.load()
    v = I.team_view(df, dept, team)
    if v["suppressed"]:
        st.warning(v["message"])
        return
    a, b = st.columns(2)
    a.metric("Responses (13 weeks)", v["count"])
    b.metric("Negative share", f"{v['negative_share']}%")
    st.markdown("#### AI summary of themes")
    try:
        st.write(I.summarise(v["df"], f"{team} team"))
    except Exception:
        st.caption("AI summary unavailable right now.")
    chart_categories(v["categories"], "Topics raised")
    db.log("manager", "viewed_team_summary", "", f"{dept}/{team}")


def dept_head_view():
    st.title("Department trends")
    if empty_db_notice():
        return
    dept = st.selectbox("Department", list(COMPANY))
    df = I.load()
    v = I.department_view(df, dept)
    if v["suppressed"]:
        st.warning(v["message"])
        return
    a, b = st.columns(2)
    a.metric("Responses (13 weeks)", v["count"])
    b.metric("Negative share", f"{v['negative_share']}%")
    st.markdown("#### AI summary of themes")
    try:
        st.write(I.summarise(v["df"], f"{dept} department"))
    except Exception:
        st.caption("AI summary unavailable right now.")
    c1, c2 = st.columns(2)
    with c1:
        chart_categories(v["categories"], "Topics raised")
    with c2:
        st.markdown("**Team comparison**")
        st.dataframe(v["teams"], hide_index=True, use_container_width=True)
        st.caption("Manager-level feedback is included in the department total only, never by team. " + I.COMMITTEE_NOTE)
        if v.get("note"):
            st.caption(v["note"])
    tr = I.monthly_trend(I.line_view(df[df["department"] == dept]), by="department")
    if not tr.empty:
        fig = px.line(tr, x="month", y="% negative", markers=True, title="Negative feedback share by month",
                      labels={"month": ""})
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    db.log("dept_head", "viewed_department", "", dept)


def leadership_view():
    st.title("Company health")
    st.caption("Company-wide patterns only. No individual notes reach this view.")
    if empty_db_notice():
        return
    df = I.load()
    recent = I.recent_weeks(df, 13)
    a, b, c = st.columns(3)
    a.metric("Health score (last 13 weeks)", f"{I.health_score(recent)}/100",
             delta=I.health_score(recent) - I.health_score(I.recent_weeks(df, 26).loc[lambda d: ~d.index.isin(recent.index)]))
    b.metric("Responses (13 weeks)", I.rounded(len(recent)))
    c.metric("Escalated to committees", I.rounded(int((recent["route"] != "hr").sum())))
    c1, c2 = st.columns(2)
    with c1:
        top = I.category_counts(recent).head(5)
        chart_categories(top, "Top 5 issues")
    with c2:
        st.markdown("**Rising and falling issues**")
        st.dataframe(I.rising_issues(df).rename(columns={"department": "Department"}), hide_index=True,
                     use_container_width=True, height=300)
    tr = I.monthly_trend(df, by="division")
    fig = px.line(tr, x="month", y="% negative", color="division", markers=True,
                  title="Negative feedback share by division", labels={"month": ""})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### AI summary for leadership")
    try:
        st.write(I.summarise(recent, "the whole company"))
    except Exception:
        st.caption("AI summary unavailable right now.")


# ================================================================== ADMIN

def admin_view():
    st.title("Admin")
    df = I.load()
    a, b, c, d = st.columns(4)
    a.metric("Live submissions", int((df["source"] == "live").sum()) if not df.empty else 0)
    b.metric("Demo (seed) notes", int((df["source"] == "seed").sum()) if not df.empty else 0)
    blocked = pd.DataFrame(db.query("SELECT reason, layer, COUNT(*) AS n FROM blocked GROUP BY reason, layer"))
    c.metric("Blocked submissions", int(blocked["n"].sum()) if not blocked.empty else 0)
    from app.pipeline.retrieval import handbook_index
    d.metric("Retrieval backend", handbook_index().backend)
    st.markdown("**Blocked by guardrails** (only the reason is stored, never the text)")
    st.dataframe(blocked, hide_index=True, use_container_width=True)
    if not df.empty:
        st.markdown("**Cases by queue and status**")
        st.dataframe(pd.crosstab(df["route"].map(ROUTE_LABELS), df["status"].map(db.STATUS_LABELS)),
                     use_container_width=True)
    st.markdown("**Audit log** (latest 200 actions)")
    st.dataframe(pd.DataFrame(db.query("SELECT week, role, action, ticket_id, detail FROM audit ORDER BY id DESC LIMIT 200")),
                 hide_index=True, use_container_width=True)
    st.caption(f"Settings: k = {K}; AI provider: {active_provider()}. Change them in .env or Streamlit secrets.")


# ================================================================== router

if role == "employee":
    employee_view()
elif role in ROUTE_LABELS:
    reviewer_view(role)
elif role == "manager":
    manager_view()
elif role == "dept_head":
    dept_head_view()
elif role == "leadership":
    leadership_view()
else:
    admin_view()
