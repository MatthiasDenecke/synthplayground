import numpy as np
import pytest

from synthplayground import registry
from synthplayground.algorithms.sine import SineSynth
from synthplayground.synth import Synth


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_discovered", True)


class DummySynth(Synth):
    def _render(self, num_samples):
        return np.zeros(num_samples)


def test_sine_is_discovered():
    assert "sine" in registry.available_synths()
    assert registry.get_synth("sine") is SineSynth


def test_create_synth_passes_kwargs():
    synth = registry.create_synth("sine", sample_rate=8000)
    assert isinstance(synth, SineSynth)
    assert synth.sample_rate == 8000


def test_register_and_lookup(clean_registry):
    registry.register_synth("dummy")(DummySynth)
    assert registry.available_synths() == ["dummy"]
    assert registry.get_synth("dummy") is DummySynth


def test_reregistering_same_class_is_allowed(clean_registry):
    registry.register_synth("dummy")(DummySynth)
    registry.register_synth("dummy")(DummySynth)


def test_duplicate_name_rejected(clean_registry):
    registry.register_synth("dummy")(DummySynth)

    class Other(DummySynth):
        pass

    with pytest.raises(ValueError):
        registry.register_synth("dummy")(Other)


def test_non_synth_rejected(clean_registry):
    with pytest.raises(TypeError):
        registry.register_synth("bad")(object)


def test_unknown_name(clean_registry):
    with pytest.raises(KeyError):
        registry.get_synth("nope")
