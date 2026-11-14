import math

import numpy as np
import pytest

from synthplayground import create_synth
from synthplayground.algorithms.sine import SineSynth
from synthplayground.metrics.tone import (
    OCTAVE_BANDS,
    Note,
    attack_time,
    band_levels,
    cents,
    compass,
    envelope,
    inharmonic_level,
    inharmonic_peaks,
    level_db,
    measure_note,
    measure_notes,
    note_difference,
    partial_levels,
    release_time,
    sounding_pitch,
)

SR = 22050


def tone(frequency, periods, partials=((1, 0.5),), noise=0.0, sample_rate=SR, seed=0):
    """A steady tone of known content."""
    time = np.arange(round(periods * sample_rate / frequency)) / sample_rate
    out = sum(
        level * np.sin(2 * np.pi * multiple * frequency * time + multiple)
        for multiple, level in partials
    )
    return out + noise * np.random.default_rng(seed).standard_normal(len(time))


def test_cents_and_level_db():
    assert cents(880.0, 440.0) == pytest.approx(1200)
    assert cents(440.0, 440.0 * 2 ** (1 / 12)) == pytest.approx(-100)
    assert level_db(10.0) == pytest.approx(20)
    assert level_db(0.5) == pytest.approx(-6.0206, abs=1e-4)
    assert level_db(0.0) == -math.inf


@pytest.mark.parametrize("frequency", [55.0, 440.0, 2500.0])
def test_sounding_pitch_finds_a_tone_that_is_off_the_nominal(frequency):
    rich = tone(frequency, 72, partials=((1, 0.3), (2, 0.3), (3, 0.2)), noise=0.003)
    for off in (-40, 0, 40):  # cents
        nominal = frequency * 2 ** (off / 1200)
        assert abs(cents(sounding_pitch(rich, SR, nominal), frequency)) < 0.05


def test_partial_levels():
    rich = tone(220.0, 72, partials=((1, 0.4), (2, 0.2), (3, 0.04), (5, 0.004)))
    levels = partial_levels(rich, SR, 220.0, count=6)
    np.testing.assert_allclose(levels[[0, 1, 2, 4]], [0, -6.02, -20, -40], atol=0.01)
    assert levels[3] < -120 and levels[5] < -120  # partials that are not there


def test_partial_levels_stop_below_half_the_sample_rate():
    assert len(partial_levels(tone(2000.0, 72), SR, 2000.0, count=12)) == 4  # 8 kHz < 0.45 SR
    with pytest.raises(ValueError):
        partial_levels(tone(2000.0, 72), SR, 0.46 * SR)
    with pytest.raises(ValueError):
        partial_levels(np.zeros(1000), SR, 220.0)


def test_a_periodic_tone_has_nothing_between_its_harmonics():
    rich = tone(331.7, 72, partials=((1, 0.4), (2, 0.3), (7, 0.1), (20, 0.05)))
    assert inharmonic_level(rich, SR, 331.7) < -120


@pytest.mark.parametrize("noise_db", [-30.0, -60.0])
def test_inharmonic_level_measures_noise(noise_db):
    partials = ((1, 0.4), (2, 0.2))
    power = sum(level**2 / 2 for _, level in partials)
    noise = math.sqrt(power * 10 ** (noise_db / 10))
    noisy = tone(220.0, 400, partials, noise=noise)
    assert inharmonic_level(noisy, SR, 220.0) == pytest.approx(noise_db, abs=0.5)


def test_inharmonic_level_measures_a_tone_that_does_not_belong():
    # 1e-3 of the amplitude of the fundamental, between the 3rd and 4th harmonic.
    stray = tone(220.0, 72, partials=((1, 0.5), (3.37, 0.0005)))
    assert inharmonic_level(stray, SR, 220.0) == pytest.approx(-60, abs=2)


def test_inharmonic_level_needs_a_long_and_audible_tone():
    with pytest.raises(ValueError):
        inharmonic_level(tone(220.0, 50), SR, 220.0)
    with pytest.raises(ValueError):
        inharmonic_level(np.zeros(SR), SR, 220.0)


def test_inharmonic_peaks_find_tones_in_noise():
    partials = ((1, 0.5), (2, 0.2), (3.37, 0.005), (9.6, 0.0005))  # -40 and -60 dB strays
    noisy = tone(220.0, 600, partials, noise=0.002)
    peaks = inharmonic_peaks(noisy, SR, 220.0)
    assert len(peaks) == 2
    (first, first_level), (second, second_level) = peaks
    assert first == pytest.approx(3.37 * 220.0, abs=0.5)
    assert second == pytest.approx(9.6 * 220.0, abs=0.5)
    assert first_level == pytest.approx(-40, abs=1) and second_level == pytest.approx(-60, abs=1)


def test_inharmonic_peaks_ignore_noise_even_where_a_resonance_gathers_it():
    # Noise with narrow peaks between the harmonics, as the modes of a pipe make them.
    rng = np.random.default_rng(3)
    clean = tone(220.0, 600, ((1, 0.5), (2, 0.2)))
    spectrum = np.fft.rfft(rng.standard_normal(len(clean)))
    frequency = np.fft.rfftfreq(len(clean), 1 / SR)
    for centre in (2.4 * 220, 5.5 * 220, 8.45 * 220):
        spectrum *= 1 + 30 / (1 + ((frequency - centre) / 4) ** 2)  # 30 dB peaks, 8 Hz wide
    noisy = clean + 0.0005 * np.fft.irfft(spectrum, len(clean))
    assert inharmonic_peaks(noisy, SR, 220.0) == []
    # The same peaks do stand out from the noise around them: a looser test would report them.
    assert inharmonic_level(noisy, SR, 220.0) > inharmonic_level(clean, SR, 220.0) + 50


def test_inharmonic_peaks_respect_margin_and_floor():
    noisy = tone(220.0, 600, ((1, 0.5), (3.37, 0.0005)), noise=0.0005)
    assert len(inharmonic_peaks(noisy, SR, 220.0, margin=10)) == 1
    assert inharmonic_peaks(noisy, SR, 220.0, margin=40) == []
    assert inharmonic_peaks(noisy, SR, 220.0, floor=-50) == []


def test_inharmonic_peaks_need_enough_periods_to_tell_tones_from_noise():
    with pytest.raises(ValueError, match="300 periods"):
        inharmonic_peaks(tone(220.0, 200, noise=0.001), SR, 220.0)


@pytest.mark.parametrize("frequency", [110.0, 1000.0, 2637.0])
def test_envelope_of_a_steady_tone_is_steady(frequency):
    # Also with only 8.4 samples to the period.
    rich = tone(frequency, 200, partials=((1, 0.5), (2, 0.2)))
    level = envelope(rich, SR, frequency)[round(8 * SR / frequency) :]
    assert level.max() / level.min() < 1.03
    assert level.mean() == pytest.approx(math.sqrt((0.5**2 + 0.2**2) / 2), rel=0.01)
    longer = envelope(rich, SR, frequency, periods=8)[round(8 * SR / frequency) :]
    assert longer.max() / longer.min() < 1.008


def test_envelope_follows_the_level():
    steady = tone(220.0, 200)
    step = steady * np.where(np.arange(len(steady)) < 10000, 1.0, 0.25)
    level = envelope(step, SR, 220.0)
    assert level[9000] == pytest.approx(0.5 / math.sqrt(2), rel=0.01)
    assert level[11000] == pytest.approx(0.125 / math.sqrt(2), rel=0.01)


def test_attack_and_release_time():
    steady = 0.5 / math.sqrt(2)
    full = tone(220.0, 300)
    rising = full * np.clip(np.arange(len(full)) / (0.2 * SR), 0, 1)  # linear, 0.2 s
    assert attack_time(rising, SR, 220.0, steady) == pytest.approx(0.18, abs=0.006)
    assert attack_time(rising, SR, 220.0, steady, share=0.5) == pytest.approx(0.10, abs=0.006)
    assert math.isnan(attack_time(0.5 * rising, SR, 220.0, steady))

    dying = full * np.exp(-np.arange(len(full)) / (0.05 * SR))  # to 1/e in 50 ms
    assert release_time(dying, SR, 220.0, steady) == pytest.approx(0.05 * math.log(10), abs=0.006)
    assert release_time(dying, SR, 220.0, steady, drop=40) == pytest.approx(0.23, abs=0.006)
    assert math.isnan(release_time(full, SR, 220.0, steady))


def test_band_levels():
    time = np.arange(SR) / SR
    audio = 0.5 * np.sin(2 * np.pi * 1000 * time) + 0.05 * np.sin(2 * np.pi * 130 * time)
    levels = band_levels(audio, SR)
    assert len(levels) == len(OCTAVE_BANDS) == 9
    assert levels[4] == pytest.approx(10 * math.log10(0.5**2 / 2), abs=0.01)  # 1 kHz band
    assert levels[1] == pytest.approx(10 * math.log10(0.05**2 / 2), abs=0.01)  # 125 Hz band
    assert np.all(levels[[0, 2, 3, 5, 6]] < -100)
    assert levels[8] == -math.inf  # 16 kHz: above half the sample rate
    np.testing.assert_allclose(band_levels(audio, SR, centres=[1000.0]), levels[4:5])
    # A rendering twice as loud is 6 dB up in every band it has.
    difference = band_levels(2 * audio, SR)[[1, 4]] - levels[[1, 4]]
    np.testing.assert_allclose(difference, 6.02, atol=0.01)


def test_band_levels_reject_bad_audio():
    with pytest.raises(ValueError):
        band_levels(np.zeros(0), SR)
    with pytest.raises(ValueError):
        band_levels(np.zeros((10, 2)), SR)


def test_measure_note_on_a_known_synth(additive):
    synth = additive(
        partials=((1, 0.4), (2, 0.2), (3, 0.04)), noise=0.004, attack=0.1, release=0.05, sharp=1.001
    )
    synth.render(num_samples=777)  # whatever came before must not matter
    note = measure_note(synth, 220.0)
    assert note.nominal == 220.0
    assert note.pitch == pytest.approx(220.22, abs=0.002)
    assert note.detuning == pytest.approx(cents(1.001, 1.0), abs=0.02)
    assert note.level == pytest.approx(math.sqrt((0.4**2 + 0.2**2 + 0.04**2) / 2), rel=0.01)
    assert 0.4 < note.peak <= 0.4 + 0.2 + 0.04 + 0.03  # at most all partials and noise at once
    np.testing.assert_allclose(note.partials[:3], [0, -6.02, -20], atol=0.05)
    assert all(level < -40 for level in note.partials[3:])
    assert note.inharmonic == pytest.approx(10 * math.log10(0.004**2 / note.level**2), abs=1.5)
    assert note.fluctuation < 0.02
    assert note.attack == pytest.approx(0.09, abs=0.006)
    assert note.overshoot == pytest.approx(1.0, abs=0.02)
    assert note.release == pytest.approx(0.045, abs=0.006)


def test_measure_note_passes_sound_parameters_on():
    quiet = measure_note(SineSynth(sample_rate=SR), 330.0, amplitude=0.25)
    assert quiet.level == pytest.approx(0.25 / math.sqrt(2), rel=1e-3)
    assert abs(quiet.detuning) < 0.01
    assert quiet.inharmonic < -100 and quiet.partials[1] < -100


def test_measure_note_of_an_organ_pipe():
    organ = create_synth("organ_waveguide", sample_rate=SR, tuning_spread_cents=0.0)
    note = measure_note(organ, 220.0)
    assert abs(note.detuning) < 0.5
    assert -12 < note.partials[1] < 0
    assert -70 < note.inharmonic < -30  # breath noise
    assert 5 / 220 < note.attack < 40 / 220  # some 10 to 20 periods
    assert 1.0 <= note.overshoot < 1.4
    assert 0 < note.release < 40 / 220
    assert note.fluctuation < 0.03
    # A 4' stop alone sounds an octave above the key.
    octave = measure_note(organ, 220.0, pitch=440.0, stops=(4,))
    assert octave.nominal == 440.0 and abs(octave.detuning) < 0.5


def test_measure_note_rejects_what_it_cannot_measure(additive):
    with pytest.raises(ValueError, match="periods"):
        measure_note(SineSynth(sample_rate=SR), 220.0, periods=50)
    with pytest.raises(ValueError, match="silent"):
        measure_note(additive(partials=((1, 0.0),)), 220.0)
    with pytest.raises(FloatingPointError):
        measure_note(additive(partials=((1, math.nan),)), 220.0)


def test_measure_notes(additive):
    notes = measure_notes(additive(), [110.0, 220.0, 440.0])
    assert [note.nominal for note in notes] == [110.0, 220.0, 440.0]
    assert all(abs(note.detuning) < 0.01 for note in notes)


def test_compass():
    octave = compass(110.0, 220.0)
    assert len(octave) == 13 and octave[0] == 110.0
    assert octave[-1] == pytest.approx(220.0)
    assert octave[1] / octave[0] == pytest.approx(2 ** (1 / 12))
    assert compass(110.0, 219.0)[-1] == pytest.approx(220.0 / 2 ** (1 / 12))
    assert compass(440.0, 440.0) == [440.0]
    for bad in ((0.0, 100.0), (200.0, 100.0)):
        with pytest.raises(ValueError):
            compass(*bad)


def test_note_difference():
    reference = Note(
        nominal=220.0,
        pitch=220.0,
        level=0.1,
        peak=0.3,
        partials=(0.0, -6.0, -12.0),
        inharmonic=-40.0,
        fluctuation=0.01,
        attack=0.05,
        overshoot=1.1,
        release=0.06,
    )
    note = Note(
        nominal=220.0,
        pitch=220.0 * 2 ** (3 / 1200),
        level=0.2,
        peak=0.5,
        partials=(0.0, -8.0),  # one partial fewer, as at a lower sample rate
        inharmonic=-46.0,
        fluctuation=0.02,
        attack=0.04,
        overshoot=1.0,
        release=0.08,
    )
    difference = note_difference(note, reference)
    assert difference.pitch == pytest.approx(3.0)
    assert difference.level == pytest.approx(6.02, abs=0.01)
    assert difference.partials == pytest.approx((0.0, -2.0))
    assert difference.inharmonic == pytest.approx(-6.0)
    assert difference.attack == pytest.approx(-0.01)
    assert difference.overshoot == pytest.approx(-0.1)
    assert difference.release == pytest.approx(0.02)
    same = note_difference(note, note)
    assert same.pitch == same.level == same.attack == 0 and set(same.partials) == {0.0}
