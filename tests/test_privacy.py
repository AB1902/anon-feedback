"""Unit tests for the rule layer (no API calls).  Run:  python -m pytest -q"""
from app.pipeline.privacy import anonymise, scrub


def s(text):
    return scrub(text, use_presidio=False).text


def test_roster_full_name_and_honorific():
    assert s("Priya Patel ma'am ignored me") == "[PERSON] ignored me"


def test_first_name_with_sir():
    assert s("Rahul sir shouts daily") == "[PERSON] shouts daily"


def test_contact_details_and_ids():
    out = s("Mail priya.patel@nimbustech.example, call +91 9349817734, ID NB-10018")
    assert out == "Mail [EMAIL], call [PHONE], ID [EMPLOYEE_ID]"


def test_team_with_location():
    assert s("the Pune Payments team is overloaded") == "the [TEAM] is overloaded"


def test_project_names():
    assert s("Project Falcon and Atlas Migration slipped") == "[PROJECT] and [PROJECT] slipped"


def test_no_false_positive_on_ordinary_words():
    text = "Harsh feedback on the platform; payments and data are delayed."
    assert s(text) == text


def test_final_text_has_no_placeholders():
    res = anonymise("Rahul sir from Pune Payments team, NB-10023", use_llm=False, use_presidio=False)
    assert "[" not in res.final_text.replace("[ID removed]", "")
    assert "Rahul" not in res.final_text and "NB-10023" not in res.final_text
