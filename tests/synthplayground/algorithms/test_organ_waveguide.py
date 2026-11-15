import dataclasses
import math

import numpy as np
import pytest

from synthplayground import available_synths, create_synth
from synthplayground.algorithms.organ_waveguide import WaveguideOrganSynth, WaveguidePipe
from synthplayground.organ import PRINCIPAL, PipeOrganSynth, measure_tone

SR = 22050


def blow(pipe, frequency, periods=120):
    """Steady part of the tone of a freshly blown pipe."""
    period = SR / frequency
    return pipe.render(round(periods * period), blowing=True)[round(0.7 * periods * period) :]


def test_registered():
    assert "organ_waveguide" in available_synths()
    synth = create_synth("organ_waveguide", sample_rate=SR)
    assert isinstance(synth, WaveguideOrganSynth) and isinstance(synth, PipeOrganSynth)


@pytest.mark.parametrize("frequency", [40.0, 196.0, 880.0, 2700.0])
def test_pipe_speaks_near_its_design_frequency(frequency):
    tone = blow(WaveguidePipe(frequency, SR), frequency)
    sounding, fundamental = measure_tone(tone, SR, frequency)
    # The jet pulls the pitch a little away from the resonance of the pipe.
    assert abs(1200 * math.log2(sounding / frequency)) < 40
    assert fundamental > np.sqrt(np.mean(tone**2))


def test_pipe_is_silent_until_blown_and_rings_out_afterwards():
    pipe = WaveguidePipe(220.0, SR)
    assert not pipe.render(500, blowing=False).any()
    assert pipe.silent
    pipe.render(4000, blowing=True)
    assert not pipe.silent
    tail = pipe.render(2 * SR, blowing=False)
    assert np.abs(tail[:200]).max() > 1e-3
    assert np.abs(tail[-200:]).max() < 1e-6
    assert pipe.silent


def test_rendering_in_blocks_is_seamless():
    whole = WaveguidePipe(330.0, SR, noise_seed=3).render(2500, blowing=True)
    pipe = WaveguidePipe(330.0, SR, noise_seed=3)
    pieces = np.concatenate([pipe.render(n, blowing=True) for n in (1, 0, 499, 2000)])
    np.testing.assert_allclose(pieces, whole, rtol=0, atol=1e-12)


def test_breath_noise_depends_on_the_seed_but_the_tone_does_not():
    one, other = (blow(WaveguidePipe(440.0, SR, noise_seed=seed), 440.0) for seed in (1, 2))
    difference = np.sqrt(np.mean((one - other) ** 2)) / np.sqrt(np.mean(one**2))
    assert 1e-4 < difference < 0.05


def test_pitch_sags_while_the_wind_dies_away():
    frequency = 220.0
    period = SR / frequency
    pipe = WaveguidePipe(frequency, SR)
    steady = blow(pipe, frequency)
    release = pipe.render(round(12 * period), blowing=False)
    sounding, _ = measure_tone(steady, SR, frequency)
    sagging, _ = measure_tone(release, SR, sounding)
    assert sagging < sounding


def test_a_struck_pipe_rings_at_its_design_frequency_and_dies_away():
    pipe = WaveguidePipe(220.0, SR)
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
    gentle, firmer = WaveguidePipe(220.0, SR), WaveguidePipe(220.0, SR)
    gentle.strike(1e-5)
    firmer.strike(1e-4)
    loud, soft = firmer.render(2000, blowing=False), gentle.render(2000, blowing=False)
    np.testing.assert_allclose(loud, 10 * soft, rtol=0, atol=1e-3 * np.abs(loud).max())


def test_pipe_takes_a_voicing():
    def speak(voicing):
        return blow(WaveguidePipe(220.0, SR, voicing=voicing), 220.0)

    usual = speak(PRINCIPAL)
    np.testing.assert_array_equal(usual, blow(WaveguidePipe(220.0, SR), 220.0))  # the default
    # A mouth that loses more leaves less tone for the same jet.
    damped = speak(dataclasses.replace(PRINCIPAL, mouth_loss=3 * PRINCIPAL.mouth_loss))
    assert not np.allclose(damped, usual)
    # No turbulence, no breath noise: two noise seeds then sound the same.
    silent = dataclasses.replace(PRINCIPAL, turbulence=0.0)
    one, other = (blow(WaveguidePipe(220.0, SR, seed, silent), 220.0) for seed in (1, 2))
    np.testing.assert_array_equal(one, other)
