import pytest

from review_classifier import data


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Network access must be explicitly simulated in tests')
    monkeypatch.setattr(data, 'urlopen', denied)
