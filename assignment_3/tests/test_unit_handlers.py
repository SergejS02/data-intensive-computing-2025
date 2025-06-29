"""Pure-Python unit tests for each handler."""
from lambdas.preprocess.handler import preprocess as _pre
from lambdas.profanity.handler  import pf
from lambdas.sentiment.handler  import sid

def test_preprocess_basic():
    txt = "This Product, IS absolutely Terrible!!"
    cleaned = _pre(txt)
    assert "terrible" in cleaned
    assert "product" in cleaned
    assert "is" not in cleaned  # stopword removed

def test_profanity_filter():
    assert pf.is_profane("shit happens") is True
    assert pf.is_profane("flowers are nice") is False
    assert pf.is_profane("this is bad") is False  # edge case

def test_sentiment_scoring():
    assert sid.polarity_scores("I love it")["compound"] > 0.25
    assert sid.polarity_scores("I hate it")["compound"] < -0.25