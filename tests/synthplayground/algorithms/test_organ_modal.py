import dataclasses
import math

import numpy as np
import pytest

from synthplayground import available_synths, create_synth
from synthplayground.algorithms.organ_modal import MAX_MODES, ModalOrganSynth, ModalPipe
from synthplayground.organ import PRINCIPAL, PipeOrganSynth, measure_tone

SR = 22050


def blow(pipe, frequency, periods=120):
    """Steady part of the tone of a freshly blown pipe."""
    period = SR / frequency
    return pipe.render(round(periods * period), blowing=True)[round(0.7 * periods * period) :]


def test_registered():
    assert "organ_modal" in available_synths()
    synth = create_synth("organ_modal", sample_rate=SR)
    assert isinstance(synth, ModalOrganSynth) and isinstance(synth, PipeOrganSynth)


def test_modes_are_stretched_harmonics():
    frequencies = ModalPipe(110.0, SR).mode_frequencies
    assert frequencies[0] == pytest.approx(110.0, rel=1e-6)
    stretch = frequencies / (110.0 * np.arange(1, len(frequencies) + 1))
    assert np.all(np.diff(stretch) > 0)  # ever sharper of the harmonics ...
    assert 1.0 < stretch[1] < 1.03 and stretch[-1] < 1.2  # ... but not by much


def test_higher_modes_die_away_faster():
    decay = ModalPipe(110.0, SR).mode_decay_times
    assert np.all(np.diff(decay) < 0)
    assert 0.02 < decay[0] < 0.5  # seconds: tens of periods


def test_number_of_modes_is_bounded_by_pipe_width_and_sample_rate():
    bass = ModalPipe(40.0, SR).mode_frequencies
    treble = ModalPipe(2000.0, SR).mode_frequencies
    assert len(treble) < len(bass) <= MAX_MODES
    assert treble[-1] < 0.45 * SR < 2000.0 * (len(treble) + 1) * 1.2
    assert bass[-1] < 0.3 * SR  # a wide pipe stops resonating long before Nyquist
    assert len(ModalPipe(2000.0, 2 * SR).mode_frequencies) > len(treble)


@pytest.mark.parametrize("frequency", [40.0, 196.0, 880.0, 2700.0])
def test_pipe_speaks_near_its_design_frequency(frequency):
    tone = blow(ModalPipe(frequency, SR), frequency)
    sounding, fundamental = measure_tone(tone, SR, frequency)
    # The jet pulls the pitch a little away from the first mode of the pipe.
    assert abs(1200 * math.log2(sounding / frequency)) < 40
    assert fundamental > np.sqrt(np.mean(tone**2))


def test_jet_locks_the_stretched_modes_into_harmonics():
    frequency = 110.0
    pipe = ModalPipe(frequency, SR)
    tone = blow(pipe, frequency, periods=200)
    sounding, _ = measure_tone(tone, SR, frequency)
    # The second partial sits on twice the sounding pitch, not on the second mode.
    second, _ = measure_tone(tone, SR, 2 * sounding)
    assert second == pytest.approx(2 * sounding, rel=1e-4)
    assert abs(second - pipe.mode_frequencies[1]) > 10 * abs(second - 2 * sounding)


def test_pipe_radiates_no_steady_pressure():
    tone = blow(ModalPipe(220.0, SR), 220.0)
    assert abs(np.mean(tone)) < 0.01 * np.sqrt(np.mean(tone**2))


def test_pipe_is_silent_until_blown_and_rings_out_afterwards():
    pipe = ModalPipe(220.0, SR)
    assert not pipe.render(500, blowing=False).any()
    assert pipe.silent
    pipe.render(4000, blowing=True)
    assert not pipe.silent
    tail = pipe.render(2 * SR, blowing=False)
    assert np.abs(tail[:200]).max() > 1e-3
    assert np.abs(tail[-200:]).max() < 1e-6
    assert pipe.silent


def test_rendering_in_blocks_is_seamless():
    whole = ModalPipe(330.0, SR, noise_seed=3).render(2500, blowing=True)
    pipe = ModalPipe(330.0, SR, noise_seed=3)
    pieces = np.concatenate([pipe.render(n, blowing=True) for n in (1, 0, 499, 2000)])
    np.testing.assert_allclose(pieces, whole, rtol=0, atol=1e-12)


def test_breath_noise_depends_on_the_seed_but_the_tone_does_not():
    one, other = (blow(ModalPipe(440.0, SR, noise_seed=seed), 440.0) for seed in (1, 2))
    difference = np.sqrt(np.mean((one - other) ** 2)) / np.sqrt(np.mean(one**2))
    assert 1e-4 < difference < 0.05


def test_a_pipe_too_high_for_the_sample_rate_is_rejected():
    with pytest.raises(ValueError):
        ModalPipe(0.46 * SR, SR)


def test_a_struck_pipe_rings_at_its_design_frequency_and_dies_away():
    pipe = ModalPipe(220.0, SR)
    pipe.strike(1e-3)
    assert not pipe.silent
    ringing = pipe.render(SR, blowing=False)
    assert 1e-6 < np.abs(ringing).max() < 1e-2
    # The fundamental outlasts the other resonances.
    late = ringing[round(0.1 * SR) : round(0.3 * SR)]
    sounding, _ = measure_tone(late, SR, 220.0)
    assert abs(1200 * math.log2(sounding / 220.0)) < 3
    assert np.abs(ringing[-2000:]).max() < 1e-3 * np.abs(ringing).max()
    pipe.render(2 * SR, blowing=False)
    assert pipe.silent


def test_ringing_scales_with_the_strike_while_it_is_gentle():
    gentle, firmer = ModalPipe(220.0, SR), ModalPipe(220.0, SR)
    gentle.strike(1e-5)
    firmer.strike(1e-4)
    loud, soft = firmer.render(2000, blowing=False), gentle.render(2000, blowing=False)
    np.testing.assert_allclose(loud, 10 * soft, rtol=0, atol=1e-3 * np.abs(loud).max())


def test_pipe_takes_a_voicing():
    def speak(voicing):
        return blow(ModalPipe(220.0, SR, voicing=voicing), 220.0)

    usual = speak(PRINCIPAL)
    np.testing.assert_array_equal(usual, blow(ModalPipe(220.0, SR), 220.0))  # the default
    # A mouth that loses more leaves less tone for the same jet.
    damped = speak(dataclasses.replace(PRINCIPAL, mouth_loss=3 * PRINCIPAL.mouth_loss))
    assert not np.allclose(damped, usual)
    # No turbulence, no breath noise: two noise seeds then sound the same.
    silent = dataclasses.replace(PRINCIPAL, turbulence=0.0)
    one, other = (blow(ModalPipe(220.0, SR, seed, silent), 220.0) for seed in (1, 2))
    np.testing.assert_array_equal(one, other)
