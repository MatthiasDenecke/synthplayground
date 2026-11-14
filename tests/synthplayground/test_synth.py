import numpy as np
import pytest

from synthplayground.synth import Synth


class ConstantSynth(Synth):
    def _render(self, num_samples):
        return np.full(num_samples, 0.5)


class ParamSynth(Synth):
    def _render(self, num_samples, level=0.0):
        return np.full(num_samples, level)


class WrongLengthSynth(Synth):
    def _render(self, num_samples):
        return np.zeros(num_samples + 1)


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        Synth()


def test_render_returns_float32_of_requested_length():
    out = ConstantSynth().render(num_samples=100)
    assert out.dtype == np.float32
    assert out.shape == (100,)
    np.testing.assert_allclose(out, 0.5)


def test_params_are_passed_through():
    synth = ParamSynth(sample_rate=8000)
    np.testing.assert_allclose(synth.render(num_samples=10, level=0.25), 0.25)
    out = synth.render(duration=0.5, level=-0.5)
    assert out.shape == (4000,)
    np.testing.assert_allclose(out, -0.5)


def test_render_zero_samples():
    assert ConstantSynth().render(num_samples=0).shape == (0,)


def test_render_rejects_negative_length():
    with pytest.raises(ValueError):
        ConstantSynth().render(num_samples=-1)


def test_render_checks_subclass_output_shape():
    with pytest.raises(RuntimeError):
        WrongLengthSynth().render(num_samples=10)


def test_duration_uses_sample_rate():
    assert ConstantSynth(sample_rate=8000).render(duration=0.5).shape == (4000,)


def test_render_rejects_negative_duration():
    with pytest.raises(ValueError):
        ConstantSynth().render(duration=-0.1)


@pytest.mark.parametrize("sample_rate", [0, -44100])
def test_invalid_sample_rate(sample_rate):
    with pytest.raises(ValueError):
        ConstantSynth(sample_rate=sample_rate)


def test_duration_rounds_to_nearest_sample():
    assert ConstantSynth(sample_rate=1000).render(duration=0.0016).shape == (2,)


@pytest.mark.parametrize("kwargs", [{}, {"num_samples": 10, "duration": 0.1}])
def test_exactly_one_length_argument_required(kwargs):
    with pytest.raises(ValueError):
        ConstantSynth().render(**kwargs)


def test_length_must_be_keyword():
    with pytest.raises(TypeError):
        ConstantSynth().render(10)
