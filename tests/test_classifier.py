"""Classifier tests (no API calls). Need a trained model: python -m scripts.train_classifier"""
import pytest

from app.pipeline import classifier

pytestmark = pytest.mark.skipif(classifier.load() is None, reason="classifier not trained")


def test_predicts_a_known_category_with_probability():
    p = classifier.predict("My salary increment was delayed again and the bonus is still not paid.")
    assert p.category == "compensation"
    assert 0 <= p.confidence <= 1 and 0 <= p.escalate_prob <= 1


def test_harassment_crosses_the_escalation_threshold():
    p = classifier.predict("He keeps making unwelcome comments about my body and sends me messages at night.")
    assert p.escalate and p.committee_category == "harassment"


def test_routine_note_stays_below_threshold():
    p = classifier.predict("The cafeteria coffee machine on floor 3 has been broken for a week.")
    assert not p.escalate
