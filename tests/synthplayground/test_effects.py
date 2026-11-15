import numpy as np
import pytest

from synthplayground.effects import church_impulse_response, church_reverb

SR = 16000


def decay_time(response, sample_rate):
    """RT60 from the slope of the backward-integrated energy between -5 and -25 dB."""
    energy = np.cumsum(response[::-1] ** 2)[::-1]
    level = 10 * np.log10(energy / energy[0])
    start, stop = np.argmax(level < -5), np.argmax(level < -25)
    return 3 * (stop - start) / sample_rate


def band(signal, sample_rate, low, high):
    spectrum = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    spectrum[(frequency < low) | (frequency >= high)] = 0
    return np.fft.irfft(spectrum, len(signal))


def test_impulse_response_is_normalised_and_reproducible():
    response = church_impulse_response(SR, rt60=2.0)
    assert np.sum(response**2) == pytest.approx(1.0)
    np.testing.assert_array_equal(response, church_impulse_response(SR, rt60=2.0))
    assert not np.array_equal(response, church_impulse_response(SR, rt60=2.0, seed=1))


@pytest.mark.parametrize("rt60", [2.0, 4.0])
def test_reverberation_time(rt60):
    response = church_impulse_response(SR, rt60=rt60)
    mid = band(response, SR, 350, 1400)  # the octaves around 500 Hz and 1 kHz
    assert decay_time(mid, SR) == pytest.approx(rt60, rel=0.2)


def test_treble_dies_away_faster_than_the_middle():
    response = church_impulse_response(SR, rt60=3.0)
    assert decay_time(band(response, SR, 4000, 8000), SR) < 0.6 * decay_time(
        band(response, SR, 350, 700), SR
    )


def test_reverberation_sets_in_after_the_predelay():
    response = church_impulse_response(SR, rt60=2.0, predelay=0.05)
    delay = round(0.05 * SR)
    assert not response[:delay].any()
    assert response[delay : delay + SR // 10].any()
    # ... and builds up rather than starting at full level.
    assert np.abs(response[delay : delay + 16]).max() < 0.2 * np.abs(response).max()


def test_reverb_adds_a_tail():
    click = np.zeros(SR // 10)
    click[0] = 1.0
    out = church_reverb(click, SR, rt60=2.0, wet=0.5)
    assert out.dtype == np.float32 and out.ndim == 1
    assert len(out) == len(click) + len(church_impulse_response(SR, rt60=2.0)) - 1
    assert out[0] == pytest.approx(0.5)  # direct sound
    assert np.abs(out[len(click) : len(click) + SR]).max() > 1e-4  # still ringing


def test_dry_and_wet_extremes():
    signal = np.random.default_rng(1).standard_normal(2000)
    dry = church_reverb(signal, SR, rt60=1.0, wet=0.0)
    np.testing.assert_allclose(dry[:2000], signal, atol=1e-6)
    assert not dry[2000:].any()
    wet = church_reverb(signal, SR, rt60=1.0, wet=1.0)
    response = church_impulse_response(SR, rt60=1.0)
    np.testing.assert_allclose(wet, np.convolve(signal, response), atol=1e-4)


def test_reverb_of_nothing_is_nothing():
    assert church_reverb(np.zeros(0), SR).shape == (0,)


@pytest.mark.parametrize(
    "kwargs", [{"wet": -0.1}, {"wet": 1.5}, {"rt60": 0.0}, {"predelay": -0.01}]
)
def test_invalid_reverb_parameters(kwargs):
    with pytest.raises(ValueError):
        church_reverb(np.zeros(10), SR, **kwargs)


def test_reverb_rejects_multichannel_audio():
    with pytest.raises(ValueError):
        church_reverb(np.zeros((10, 2)), SR)
