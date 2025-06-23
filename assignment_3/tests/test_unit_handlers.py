"""Pure-Python unit tests for each handler."""
from lambdas.preprocess.handler import preprocess as _pre
from lambdas.profanity.handler  import pf
from lambdas.sentiment.handler  import sid

def test_preprocess_basic():
    txt = "This Product, IS absolutely Terrible!!"
    cleaned = _pre(txt)
    assert "terrible" in cleaned and "product" in cleaned
    assert "is" not in cleaned                      # stop-word removed

def test_profanity_filter():
    bad  = "shit happens"
    good = "flowers are nice"
    assert pf.is_profane(bad)  is True
    assert pf.is_profane(good) is False

def test_sentiment_scoring():
    pos = sid.polarity_scores("I love it")["compound"]
    neg = sid.polarity_scores("I hate it")["compound"]
    assert pos > 0.25 and neg < -0.25
