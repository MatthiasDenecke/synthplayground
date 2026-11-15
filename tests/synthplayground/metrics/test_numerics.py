import dataclasses
import math

import pytest

from synthplayground import create_synth, get_synth
from synthplayground.algorithms.sine import SineSynth
from synthplayground.metrics.numerics import (
    aliasing_level,
    convergence,
    inharmonic_tones,
    stability,
)
from synthplayground.metrics.tone import compass
from synthplayground.organ import PRINCIPAL

SR = 22050
ORGANS = ["organ_waveguide", "organ_modal"]
SILENT_JET = dataclasses.replace(PRINCIPAL, turbulence=0.0)


def organ(name, sample_rate=SR, **kwargs):
    return create_synth(name, sample_rate=sample_rate, tuning_spread_cents=0.0, **kwargs)


# --- aliasing ----------------------------------------------------------------------


def test_aliasing_level_of_known_synths(additive):
    assert aliasing_level(SineSynth(sample_rate=SR), 440.0) < -120
    # A partial at 3.3 times the pitch, 1e-3 of the fundamental: not a harmonic.
    folded = additive(partials=((1, 0.5), (2, 0.2), (3.3, 0.0005)))
    assert aliasing_level(folded, 440.0) == pytest.approx(-60, abs=2.5)


def plain(name, **kwargs):
    """The organ without oversampling: every pipe at the sample rate itself."""
    never = type("Plain", (get_synth(name),), {"samples_per_period": 0.0})
    return never(sample_rate=SR, tuning_spread_cents=0.0, **kwargs)


@pytest.mark.parametrize("name", ORGANS)
def test_without_oversampling_aliasing_grows_towards_the_top(name):
    silent = plain(name, voicing=SILENT_JET)
    middle, top = aliasing_level(silent, 220.0), aliasing_level(silent, 2637.0)
    assert middle < -60
    assert top > middle + 15
    # With breath noise the same measurement mostly shows the noise.
    assert aliasing_level(organ(name), 220.0) > middle + 15


@pytest.mark.parametrize("name", ORGANS)
def test_aliasing_of_the_organs_stays_below_70_db(name):
    silent = organ(name, voicing=SILENT_JET)
    # Just below where oversampling sets in, where its factor steps up, and at the top.
    limit = SR / silent.samples_per_period
    pitches = [0.97 * limit, 1.03 * limit, 1.97 * limit, 2.03 * limit, 1318.5, 2637.0, SR / 8]
    assert [silent.oversampling(pitch) for pitch in pitches[:4]] == [1, 2, 2, 3]
    levels = [aliasing_level(silent, pitch) for pitch in pitches]
    assert max(levels) < -70, dict(zip(pitches, levels))


def test_inharmonic_tones_of_known_synths(additive):
    assert inharmonic_tones(additive(noise=0.001), 440.0) == []
    stray = additive(partials=((1, 0.5), (3.3, 0.005)), noise=0.001)
    ((frequency, level),) = inharmonic_tones(stray, 440.0)
    assert frequency == pytest.approx(3.3 * 440.0, abs=3)
    assert level == pytest.approx(-40, abs=1)
    assert inharmonic_tones(stray, 440.0, margin=45) == []


@pytest.mark.parametrize("name", ORGANS)
def test_without_oversampling_the_organs_have_folded_partials(name):
    pitch = 2637.0
    tones = inharmonic_tones(plain(name), pitch)
    assert tones and tones[0][1] < -20
    # Partial n above half the sample rate is mirrored to SR - n * pitch (and so on).
    folded = []
    for n in range(5, 40):
        frequency = (n * pitch) % SR
        folded.append(min(frequency, SR - frequency))
    for frequency, _ in tones[:3]:
        assert min(abs(frequency - alias) for alias in folded) < 6


@pytest.mark.parametrize("name", ORGANS)
@pytest.mark.parametrize("pitch", [220.0, 1318.5, 2637.0])
def test_no_tones_stand_out_between_the_harmonics_of_the_organs(name, pitch):
    assert inharmonic_tones(organ(name), pitch) == []


def test_inharmonic_tones_need_a_long_note():
    with pytest.raises(ValueError, match="periods"):
        inharmonic_tones(SineSynth(sample_rate=SR), 440.0, periods=300)


# --- convergence -------------------------------------------------------------------


def test_convergence_of_known_synths(additive):
    assert all(
        difference.pitch == pytest.approx(0, abs=1e-3)
        and difference.level == pytest.approx(0, abs=1e-3)
        for difference in convergence(lambda rate: SineSynth(sample_rate=rate), 440.0)
    )

    def coarse(rate):
        """Second partial 6 dB too strong and everything 2 cents sharp at 22.05 kHz."""
        wrong = rate == 22050
        partials = ((1, 0.5), (2, 0.2 if wrong else 0.1))
        return additive(sample_rate=rate, partials=partials, sharp=2 ** (2 / 1200) if wrong else 1)

    low, middle = convergence(coarse, 440.0, (22050, 44100, 88200))
    assert low.pitch == pytest.approx(2.0, abs=0.01)
    assert low.partials[1] == pytest.approx(6.02, abs=0.05)
    assert middle.pitch == pytest.approx(0, abs=0.01)
    assert middle.partials[1] == pytest.approx(0, abs=0.05)


@pytest.mark.parametrize("name", ORGANS)
def test_the_organs_converge(name):
    (difference,) = convergence(lambda rate: organ(name, rate), 220.0, (22050, 44100))
    assert abs(difference.pitch) < 0.2
    assert abs(difference.level) < 0.1
    assert max(abs(level) for level in difference.partials[:4]) < 1.5
    assert abs(difference.attack) < 0.01
    assert abs(difference.inharmonic) < 1.5  # the breath noise does not depend on the rate


@pytest.mark.parametrize("rates", [(44100,), (88200, 44100), (44100, 88200, 22050)])
def test_convergence_needs_rising_sample_rates(rates):
    with pytest.raises(ValueError):
        convergence(lambda rate: SineSynth(sample_rate=rate), 440.0, rates)


# --- stability ---------------------------------------------------------------------


def test_stability_of_a_well_behaved_synth(additive):
    checks = stability(additive(release=0.01), [110.0, 440.0, 1760.0])
    assert [check.frequency for check in checks] == [110.0, 440.0, 1760.0]
    for check in checks:
        assert check.ok and check.problem == ""
        assert abs(check.detuning) < 0.01 and check.fluctuation < 0.01
        assert check.peak == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize(
    "fault, problem",
    [
        ({"sharp": 2 ** (80 / 1200)}, "sounds +80 cents off"),
        ({"warble": 0.1}, "level varies by"),
        ({"partials": ((1, 1.2),)}, "reaches full scale"),
        ({"sustain": True}, "does not die away"),
        ({"highest": 300.0}, "ValueError: 440.0 Hz is too high"),
        ({"partials": ((1, 0.0),)}, "ValueError: the note is silent"),
        ({"partials": ((1, math.nan),)}, "FloatingPointError"),
    ],
)
def test_stability_names_what_is_wrong(additive, fault, problem):
    (check,) = stability(additive(**fault), [440.0])
    assert not check.ok
    assert problem in check.problem


def test_stability_judges_each_pitch_on_its_own(additive):
    low, high = stability(additive(highest=300.0), [220.0, 440.0])
    assert low.ok and not high.ok
    assert math.isnan(high.detuning) and math.isnan(high.fluctuation) and math.isnan(high.peak)


def test_stability_limits_can_be_set(additive):
    synth = additive(sharp=2 ** (10 / 1200), warble=0.01)
    assert stability(synth, [440.0])[0].ok
    assert "cents off" in stability(synth, [440.0], max_detuning=5.0)[0].problem
    assert "level varies" in stability(synth, [440.0], max_fluctuation=0.01)[0].problem


@pytest.mark.parametrize("name", ORGANS)
def test_the_organs_are_stable_over_their_compass(name):
    octaves = compass(41.2, SR / 8)[::6]  # every tritone from 16' E to the top
    checks = stability(organ(name), octaves)
    assert len(checks) == len(octaves) >= 12
    assert all(check.ok for check in checks), [check for check in checks if not check.ok]
    # A chorus is three pipes; a key whose 2' pipe is too high is refused, which is reported.
    assert stability(organ(name), [220.0], stops=(8, 4, 2))[0].ok
    assert "ValueError" in stability(organ(name), [1760.0], stops=(8, 4, 2))[0].problem
