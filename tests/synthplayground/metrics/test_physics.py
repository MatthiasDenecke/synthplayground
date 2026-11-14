import math

import numpy as np
import pytest

from synthplayground.algorithms.organ_modal import ModalPipe
from synthplayground.algorithms.organ_waveguide import WaveguidePipe
from synthplayground.metrics.physics import (
    PowerLaw,
    Resonance,
    ResonanceDeviation,
    detailed_resonances,
    ideal_resonances,
    measure_resonances,
    power_law,
    resonance_deviations,
)
from synthplayground.metrics.tone import cents
from synthplayground.organ import PRINCIPAL, FluePipe, PipeGeometry

SR = 44100


class Bells(FluePipe):
    """A stand-in pipe that rings at exactly known resonances when struck.

    Like a pipe it radiates through a high-pass, which tilts the spectrum but
    must not shift the resonances that are measured.
    """

    def __init__(self, frequencies, decay_times, sample_rate=SR):
        rate = -1 / np.asarray(decay_times) + 2j * np.pi * np.asarray(frequencies)
        self.poles = np.exp(rate / sample_rate)
        self.state = np.zeros(len(frequencies), dtype=complex)
        self.reflected = 0.0

    def strike(self, amount):
        self.state += amount

    @property
    def silent(self):
        return not self.state.any()

    def render(self, num_samples, blowing):
        out = np.empty(num_samples)
        for i in range(num_samples):
            self.state *= self.poles
            wave = self.state.real.sum()
            self.reflected += 0.3 * (wave - self.reflected)
            out[i] = wave - self.reflected
        return out


class Dead(FluePipe):
    silent = True

    def strike(self, amount):
        pass

    def render(self, num_samples, blowing):
        return np.zeros(num_samples)


# --- the references ----------------------------------------------------------------


def test_detailed_resonances_are_stretched_and_ever_shorter_lived():
    resonances = detailed_resonances(110.0, 8)
    assert [resonance.number for resonance in resonances] == list(range(1, 9))
    assert resonances[0].frequency == pytest.approx(110.0, rel=1e-9)
    stretch = [resonance.frequency / (110.0 * resonance.number) for resonance in resonances]
    assert all(b > a for a, b in zip(stretch, stretch[1:])) and 1.0 < stretch[-1] < 1.05
    decays = [resonance.decay_time for resonance in resonances]
    assert all(b < a for a, b in zip(decays, decays[1:])) and 0.05 < decays[0] < 0.5


def test_detailed_resonances_are_what_the_modal_pipe_is_built_from():
    pipe = ModalPipe(261.63, SR)
    resonances = detailed_resonances(261.63, 5)
    np.testing.assert_allclose(
        [resonance.frequency for resonance in resonances], pipe.mode_frequencies[:5], rtol=1e-9
    )
    np.testing.assert_allclose(
        [resonance.decay_time for resonance in resonances], pipe.mode_decay_times[:5], rtol=1e-9
    )


def test_detailed_resonances_depend_on_the_voicing():
    import dataclasses

    lossy = dataclasses.replace(PRINCIPAL, mouth_loss=2 * PRINCIPAL.mouth_loss)
    assert detailed_resonances(220.0, 1, lossy)[0].decay_time < (
        detailed_resonances(220.0, 1)[0].decay_time
    )
    plain = dataclasses.replace(PRINCIPAL, mouth_correction=0.01)  # hardly any mouth
    assert detailed_resonances(220.0, 4, plain)[3].frequency < (
        detailed_resonances(220.0, 4)[3].frequency
    )


def test_detailed_resonances_end_where_the_pipe_is_too_wide_for_them():
    # A treble pipe is wide for its length: few resonances before ka = 3.5.
    assert len(detailed_resonances(110.0, 100)) > len(detailed_resonances(3520.0, 100)) > 2
    assert len(detailed_resonances(110.0, 3)) == 3


def test_ideal_resonances_are_harmonics_damped_by_walls_and_radiation():
    resonances = ideal_resonances(110.0, 6)
    assert [resonance.frequency for resonance in resonances] == [110.0 * n for n in range(1, 7)]
    geometry = PipeGeometry.principal(110.0)
    ka = 2 * math.pi * 110.0 * geometry.radius / 343.0
    loss = geometry.wall_loss(110.0) + ka**2  # per round trip of 1 / 110 s
    assert resonances[0].decay_time == pytest.approx(1 / (110.0 * loss))
    # Without a mouth the fundamental rings longer than that of the real pipe.
    assert resonances[0].decay_time > 1.5 * detailed_resonances(110.0, 1)[0].decay_time


def test_ideal_resonances_stop_where_the_radiation_formula_does():
    geometry = PipeGeometry.principal(1760.0)
    resonances = ideal_resonances(1760.0, 50)
    limit = 343.0 / (2 * math.pi * geometry.radius)  # ka = 1
    assert 1 <= len(resonances) < 50
    assert resonances[-1].frequency < limit <= resonances[-1].frequency + 1760.0


# --- measuring ---------------------------------------------------------------------


def test_measure_resonances_recovers_known_resonances():
    frequencies = [200.0, 402.0, 607.0, 815.0, 1026.0, 1240.0]
    decay_times = [0.08, 0.05, 0.03, 0.02, 0.014, 0.01]
    measured = measure_resonances(Bells(frequencies, decay_times), SR, 200.0, count=6)
    assert [resonance.number for resonance in measured] == [1, 2, 3, 4, 5, 6]
    for resonance, frequency, decay_time in zip(measured, frequencies, decay_times):
        assert abs(cents(resonance.frequency, frequency)) < 0.05
        assert resonance.decay_time == pytest.approx(decay_time, rel=0.01)


def test_measure_resonances_finds_only_as_many_as_asked_for_or_as_there_are():
    frequencies, decay_times = [200.0, 402.0, 607.0], [0.08, 0.05, 0.03]
    assert len(measure_resonances(Bells(frequencies, decay_times), SR, 200.0, count=2)) == 2
    assert len(measure_resonances(Bells(frequencies, decay_times), SR, 200.0, count=8)) == 3


def test_measure_resonances_of_the_modal_pipe_match_its_modes():
    pipe = ModalPipe(220.0, SR)
    measured = measure_resonances(pipe, SR, 220.0, count=8)
    assert len(measured) == 8
    for resonance, frequency, decay_time in zip(
        measured, pipe.mode_frequencies, pipe.mode_decay_times
    ):
        assert abs(cents(resonance.frequency, frequency)) < 0.1
        assert resonance.decay_time == pytest.approx(decay_time, rel=0.02)


@pytest.mark.parametrize("frequency", [110.0, 880.0])
def test_measure_resonances_of_the_waveguide_pipe_are_close_to_theory(frequency):
    measured = measure_resonances(WaveguidePipe(frequency, SR), SR, frequency, count=6)
    assert len(measured) == 6
    deviations = resonance_deviations(measured, detailed_resonances(frequency, 6))
    assert abs(deviations[0].cents) < 2  # the loop is sized for the design frequency
    assert all(abs(deviation.cents) < 25 for deviation in deviations)
    assert all(0.8 < deviation.decay_ratio < 1.25 for deviation in deviations)
    # Against the ideal pipe the resonances are stretched: physics, not error.
    ideal = resonance_deviations(measured, ideal_resonances(frequency, 6))
    assert all(b.cents > a.cents for a, b in zip(ideal, ideal[1:]))


def test_measure_resonances_needs_a_pipe_that_rings():
    with pytest.raises(ValueError, match="does not ring"):
        measure_resonances(Dead(), SR, 220.0)


def test_resonance_deviations_pair_resonances_by_number():
    measured = [Resonance(1, 101.0, 0.1), Resonance(2, 204.0, 0.05), Resonance(3, 310.0, 0.02)]
    reference = [Resonance(2, 200.0, 0.1), Resonance(1, 100.0, 0.1)]  # any order, no third
    deviations = resonance_deviations(measured, reference)
    assert deviations == [
        ResonanceDeviation(1, pytest.approx(cents(101.0, 100.0)), pytest.approx(1.0)),
        ResonanceDeviation(2, pytest.approx(cents(204.0, 200.0)), pytest.approx(0.5)),
    ]
    assert resonance_deviations(measured, []) == []


# --- scaling -----------------------------------------------------------------------


def test_power_law_recovers_exponent_and_scale():
    frequencies = [55.0, 110.0, 220.0, 440.0, 880.0]
    law = power_law(frequencies, [3.0 * f**-0.25 for f in frequencies])
    assert isinstance(law, PowerLaw)
    assert law.exponent == pytest.approx(-0.25)
    assert law.scale == pytest.approx(3.0)
    assert law.worst == pytest.approx(0.0, abs=1e-9)
    assert law(1760.0) == pytest.approx(3.0 * 1760.0**-0.25)


def test_power_law_tells_how_far_a_value_strays():
    frequencies = [100.0 * 2**i for i in range(9)]
    values = [1 / f for f in frequencies]
    values[4] *= 2  # one pipe twice as loud as the rule says: 6 dB
    law = power_law(frequencies, values)
    assert law.exponent == pytest.approx(-1.0, abs=0.01)
    assert law.worst == pytest.approx(6.02 * 8 / 9, abs=0.01)  # the fit gives way a little


@pytest.mark.parametrize(
    "frequencies, values",
    [([100.0], [1.0]), ([100.0, 200.0], [1.0]), ([100.0, 200.0], [1.0, 0.0]), ([0.0, 1.0], [1, 1])],
)
def test_power_law_rejects_bad_input(frequencies, values):
    with pytest.raises(ValueError):
        power_law(frequencies, values)
