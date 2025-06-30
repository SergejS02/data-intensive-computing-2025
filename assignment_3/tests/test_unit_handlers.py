from lambdas.preprocess.handler import preprocess as _pre
from lambdas.profanity.handler  import pf
from lambdas.sentiment.handler  import sid

#test each handler logic independently

def test_preprocess_basic():
    txt = "This Product, IS absolutely terrible!!"
    cleaned = _pre(txt)
    assert "terrible" in cleaned and "product" in cleaned
    assert "is" not in cleaned   # stop-word removed

def test_profanity_filter():
    bad  = "shit happens"
    good = "chocolate is nice"
    assert pf.is_profane(bad)  is True
    assert pf.is_profane(good) is False

def test_sentiment_scoring():
    pos = sid.polarity_scores("I love it")["compound"]
    neg = sid.polarity_scores("I hate it")["compound"]
    assert pos > 0.25 and neg < -0.25
