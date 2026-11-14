import numpy as np
import pytest

from synthplayground.algorithms.sine import SineSynth


def test_starts_at_zero_and_matches_reference():
    sr, freq = 8000, 440.0
    out = SineSynth(sample_rate=sr).render(num_samples=256, frequency=freq)
    expected = np.sin(2 * np.pi * freq * np.arange(256) / sr)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_default_parameters():
    sr = 8000
    out = SineSynth(sample_rate=sr).render(num_samples=256)
    expected = np.sin(2 * np.pi * 440.0 * np.arange(256) / sr)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_amplitude():
    out = SineSynth(sample_rate=8000).render(duration=1.0, frequency=100.0, amplitude=0.25)
    assert np.max(np.abs(out)) == pytest.approx(0.25, abs=1e-6)


def test_dominant_frequency():
    sr, freq = 8000, 1000.0
    out = SineSynth(sample_rate=sr).render(duration=1.0, frequency=freq)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.argmax(spectrum) * sr / len(out)
    assert peak_hz == pytest.approx(freq, abs=1.0)


def test_consecutive_renders_are_continuous():
    whole = SineSynth(sample_rate=8000).render(num_samples=1000, frequency=123.4)
    synth = SineSynth(sample_rate=8000)
    pieces = np.concatenate(
        [synth.render(num_samples=n, frequency=123.4) for n in (1, 299, 400, 300)]
    )
    np.testing.assert_allclose(pieces, whole, atol=1e-5)


def test_frequency_change_keeps_phase_continuous():
    sr = 8000
    synth = SineSynth(sample_rate=sr)
    first = synth.render(num_samples=100, frequency=440.0)
    second = synth.render(num_samples=100, frequency=880.0)
    phase_at_switch = 440.0 * 100 / sr
    expected = np.sin(2 * np.pi * (phase_at_switch + 880.0 * np.arange(100) / sr))
    np.testing.assert_allclose(second, expected, atol=1e-5)
    assert abs(second[0] - first[-1]) < 2 * np.pi * 880.0 / sr


def test_reset_restarts_phase():
    synth = SineSynth(sample_rate=8000)
    first = synth.render(num_samples=100)
    synth.render(num_samples=37)
    synth.reset()
    np.testing.assert_array_equal(synth.render(num_samples=100), first)


@pytest.mark.parametrize("frequency", [0.0, -1.0, 4000.0, 10000.0])
def test_invalid_frequency(frequency):
    with pytest.raises(ValueError):
        SineSynth(sample_rate=8000).render(num_samples=10, frequency=frequency)


@pytest.mark.parametrize("amplitude", [-0.1, 1.1])
def test_invalid_amplitude(amplitude):
    with pytest.raises(ValueError):
        SineSynth().render(num_samples=10, amplitude=amplitude)


def test_unknown_parameter_rejected():
    with pytest.raises(TypeError):
        SineSynth().render(num_samples=10, cutoff=1000.0)
