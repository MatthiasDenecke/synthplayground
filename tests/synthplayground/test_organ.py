import dataclasses
import math

import numpy as np
import pytest

from synthplayground import create_synth
from synthplayground.organ import (
    PRINCIPAL,
    FluePipe,
    Jet,
    Oversampled,
    PipeDoesNotSpeakError,
    PipeGeometry,
    PipeOrganSynth,
    Voicing,
    Wind,
    measure_tone,
    open_end,
    pipe_modes,
    voice,
)

SR = 22050
ORGANS = ["organ_waveguide", "organ_modal"]
VOICING = Voicing(
    mouth_correction=1.5,
    mouth_loss=0.03,
    jet_travel=0.5,
    jet_inertia=1.0,
    jet_sharpness=8.0,
    jet_offset=0.7,
    jet_swing=1.0,
    jet_drive=4.0,
    turbulence=0.001,
    turbulence_band=8.0,
)


class TonePipe(FluePipe):
    """Stand-in pipe: a sine that starts at once and fades out linearly when released."""

    FADE = 50
    built = []  # the pipes built to sound (not those built for voicing)

    def __init__(self, frequency, sample_rate, level=0.5, sounding=False):
        self.frequency, self.sample_rate, self.level = frequency, sample_rate, level
        self.position = 0
        self.fade = self.FADE
        if sounding:
            TonePipe.built.append(self)

    def render(self, num_samples, blowing):
        t = (self.position + np.arange(num_samples)) / self.sample_rate
        self.position += num_samples
        tone = self.level * np.sin(2 * np.pi * self.frequency * t)
        if blowing:
            self.fade = self.FADE
            return tone
        gains = np.clip(self.fade - np.arange(num_samples), 0, None) / self.FADE
        self.fade = max(0, self.fade - num_samples)
        return tone * gains

    @property
    def silent(self):
        return self.fade == 0

    def strike(self, amount):
        raise NotImplementedError("a stand-in pipe has no air column")


class ToneOrgan(PipeOrganSynth):
    def _build_pipe(self, design_frequency, sample_rate, noise_seed):
        return TonePipe(design_frequency, sample_rate, sounding=noise_seed != 0)


@pytest.fixture
def organ():
    TonePipe.built = []
    return ToneOrgan(sample_rate=SR, tuning_spread_cents=0.0)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def cents(frequency, reference):
    return 1200 * math.log2(frequency / reference)


def harmonic_levels(tone, sample_rate, frequency, count):
    """Amplitudes of the first ``count`` harmonics of a steady tone."""
    t = np.arange(len(tone))
    window = np.hanning(len(tone))
    return [
        2 * abs(np.dot(tone, window * np.exp(-2j * np.pi * k * frequency / sample_rate * t)))
        / window.sum()
        for k in range(1, count + 1)
    ]


# --- pipe dimensions, wind and jet -------------------------------------------------


def test_principal_scale():
    low = PipeGeometry.principal(65.4)
    assert 2 * low.radius == pytest.approx(0.1555)
    assert low.length == pytest.approx(343.0 / (2 * 65.4))
    # The diameter halves on the 16th pipe above (16 semitones).
    high = PipeGeometry.principal(65.4 * 2 ** (16 / 12))
    assert high.radius == pytest.approx(low.radius / 2)


def test_high_pipes_are_relatively_wider_and_lossier():
    low, high = PipeGeometry.principal(100.0), PipeGeometry.principal(1600.0)
    assert high.radius / high.length > low.radius / low.length
    assert high.wall_loss(1600.0) > low.wall_loss(100.0) > 0


def test_wind_fills_large_pipes_more_slowly_within_limits():
    bass, tenor, treble = (Wind.for_pipe(f, 44100) for f in (20.0, 200.0, 4000.0))
    assert 0 < bass.rise < tenor.rise < treble.rise < 1
    assert Wind.for_pipe(10.0, 44100) == bass  # clamped
    assert Wind.for_pipe(8000.0, 44100) == treble
    assert tenor.fall < tenor.rise  # the wind dies away more slowly than it sets in


def test_jet_gain_makes_up_for_the_losses_of_the_pipe():
    weak, strong = (Jet.design(VOICING, 100.0, resonance) for resonance in (5.0, 20.0))
    assert weak.gain == pytest.approx(4 * strong.gain)
    assert weak.vortex > strong.vortex > 0
    assert 0 < weak.lowpass < 1 and 0 < weak.highpass < 1


def test_jet_travel_scales_with_the_period():
    short, long = (Jet.design(VOICING, period, 10.0) for period in (50.0, 200.0))
    assert long.travel == pytest.approx(4 * short.travel, rel=0.02)
    # The filters eat into the nominal half period; what is left is pure delay.
    assert 0.2 * 200 < long.travel < 0.8 * 200
    assert len(long.delay_line()) >= long.travel / Jet.SLOWEST + 2


def test_jet_without_surplus_drive_needs_no_vortex_loss():
    lazy = dataclasses.replace(VOICING, jet_drive=1.0)
    assert Jet.design(lazy, 100.0, 10.0).vortex == 0.0


def test_open_end_reflects_low_frequencies_and_lets_high_ones_out():
    reflection, correction = open_end(0.0)
    assert reflection == 1.0 and correction == pytest.approx(0.6133)
    # At low frequencies the end radiates (ka)^2 / 2 of the amplitude.
    assert 1 - open_end(0.1)[0] == pytest.approx(0.1**2 / 2, rel=0.05)
    reflections, corrections = zip(*(open_end(ka) for ka in (0.5, 1.0, 2.0, 3.5)))
    assert all(b < a for a, b in zip(reflections, reflections[1:])) and reflections[-1] > 0
    assert all(b < a for a, b in zip(corrections, corrections[1:])) and corrections[-1] > 0


def test_pipe_modes_are_stretched_harmonics_that_die_away_ever_faster():
    modes = list(pipe_modes(PipeGeometry.principal(110.0), PRINCIPAL))
    assert [mode.number for mode in modes] == list(range(1, len(modes) + 1))
    assert modes[0].frequency == pytest.approx(110.0, rel=1e-9)
    stretch = [mode.frequency / (110.0 * mode.number) for mode in modes]
    assert all(b > a for a, b in zip(stretch, stretch[1:]))
    assert 1.0 < stretch[1] < 1.03 and stretch[-1] < 1.2
    assert all(b.loss > a.loss for a, b in zip(modes, modes[1:]))
    assert all(b.decay_time < a.decay_time for a, b in zip(modes, modes[1:]))
    assert modes[0].decay_time == pytest.approx(1 / (110.0 * modes[0].loss))
    assert 0.02 < modes[0].decay_time < 0.5  # tens of periods


def test_pipe_modes_end_where_the_pipe_is_too_wide_for_plane_waves():
    geometry = PipeGeometry.principal(440.0)
    modes = list(pipe_modes(geometry, PRINCIPAL))
    ka = [2 * math.pi * mode.frequency * geometry.radius / 343.0 for mode in modes]
    assert ka[-1] <= 3.5 < ka[-1] + ka[0] * 1.2
    # Trebles are wide for their length and have fewer such modes than basses.
    assert len(list(pipe_modes(PipeGeometry.principal(55.0), PRINCIPAL))) > len(modes)


def test_pipe_modes_follow_the_voicing_of_the_mouth():
    geometry = PipeGeometry.principal(220.0)
    modes = list(pipe_modes(geometry, PRINCIPAL))
    lossy = dataclasses.replace(PRINCIPAL, mouth_loss=PRINCIPAL.mouth_loss + 0.05)
    assert next(pipe_modes(geometry, lossy)).loss == pytest.approx(modes[0].loss + 0.05)
    # A smaller mouth correction leaves less to shrink: the modes are less stretched.
    small = list(pipe_modes(geometry, dataclasses.replace(PRINCIPAL, mouth_correction=0.2)))
    assert small[0].frequency == pytest.approx(220.0, rel=1e-9)
    assert small[3].frequency < modes[3].frequency


def test_breath_noise_per_sample_grows_with_the_sample_rate():
    # Twice the rate spreads the same random numbers over twice the band.
    usual = Jet.design(VOICING, 100.0, 10.0)
    assert usual.noise == VOICING.turbulence
    assert usual == Jet.design(VOICING, 100.0, 10.0, sample_rate=44100)
    fine = Jet.design(VOICING, 100.0, 10.0, sample_rate=4 * 44100)
    assert fine.noise == pytest.approx(2 * VOICING.turbulence)
    assert dataclasses.replace(fine, noise=usual.noise) == usual  # nothing else changes


# --- oversampling ------------------------------------------------------------------


class Playback(FluePipe):
    """Stand-in pipe that plays back a given signal, then silence."""

    def __init__(self, signal):
        self.signal, self.position, self.struck = np.asarray(signal, float), 0, []

    def render(self, num_samples, blowing):
        out = np.zeros(num_samples)
        chunk = self.signal[self.position : self.position + num_samples]
        out[: len(chunk)] = chunk
        self.position += num_samples
        return out

    @property
    def silent(self):
        return self.position >= len(self.signal)

    def strike(self, amount):
        self.struck.append(amount)


def sine(frequency, sample_rate, num_samples, amplitude=1.0):
    return amplitude * np.sin(2 * np.pi * frequency * np.arange(num_samples) / sample_rate)


@pytest.mark.parametrize("factor", [1, 2, 3, 8])
def test_oversampled_pipe_passes_the_audible_delayed_by_a_millisecond(factor):
    fine = sine(0.4 * SR, factor * SR, factor * 4000)  # 8.8 kHz at 22.05 kHz
    out = Oversampled(Playback(fine), factor).render(4000, blowing=True)
    assert out.shape == (4000,)
    np.testing.assert_allclose(out[200:], sine(0.4 * SR, SR, 4000)[200 - 48 : -48], atol=1e-3)


@pytest.mark.parametrize("factor", [2, 3, 8])
def test_oversampled_pipe_removes_what_would_fold_back(factor):
    # A partial above half the output rate would otherwise come out at SR - frequency.
    for frequency in (0.5 * SR, 0.6 * SR, 0.95 * SR, 0.49 * factor * SR):
        fine = sine(frequency, factor * SR, factor * 4000)
        out = Oversampled(Playback(fine), factor).render(4000, blowing=True)
        assert np.abs(out[200:]).max() < 10 ** (-70 / 20), frequency


def test_oversampled_pipe_renders_seamlessly_in_blocks():
    fine = np.random.default_rng(0).standard_normal(3 * 3000)
    whole = Oversampled(Playback(fine), 3).render(3000, blowing=True)
    pipe = Oversampled(Playback(fine), 3)
    pieces = np.concatenate([pipe.render(n, blowing=True) for n in (1, 0, 499, 2500)])
    np.testing.assert_allclose(pieces, whole, rtol=0, atol=1e-12)


def test_oversampled_pipe_is_silent_once_its_filter_has_run_empty():
    pipe = Oversampled(Playback(np.ones(2 * 100)), 2)
    pipe.render(100, blowing=True)
    assert not pipe.silent
    pipe.render(1, blowing=False)  # the pipe inside is done, but its last sound is on the way
    assert not pipe.silent
    tail = pipe.render(200, blowing=False)
    assert np.abs(tail[:40]).max() > 0.1 and not tail[100:].any()
    assert pipe.silent


def test_oversampled_pipe_hands_a_strike_on():
    inside = Playback(np.zeros(10))
    Oversampled(inside, 4).strike(0.25)
    assert inside.struck == [0.25]
    with pytest.raises(ValueError):
        Oversampled(inside, 0)


# --- voicing -----------------------------------------------------------------------


@pytest.mark.parametrize("frequency", [55.0, 439.3, 2000.0])
def test_measure_tone(frequency):
    t = np.arange(SR) / SR
    tone = 0.3 * np.sin(2 * np.pi * frequency * t + 1.0)
    tone += 0.2 * np.sin(2 * np.pi * 2 * frequency * t)
    tone += 0.1 * np.sin(2 * np.pi * 3 * frequency * t)
    measured, amplitude = measure_tone(tone, SR, frequency * 1.02)  # nominal is off by 34 cents
    assert abs(cents(measured, frequency)) < 0.05
    measured, amplitude = measure_tone(tone, SR, measured)
    assert amplitude == pytest.approx(0.3, rel=0.01)


def test_measure_tone_needs_enough_periods():
    with pytest.raises(ValueError):
        measure_tone(np.zeros(100), SR, 100.0)


def test_voice_tunes_and_regulates():
    # A pipe that sounds 2 % sharp of its design frequency, like a jet-pulled pipe.
    design, gain = voice(lambda f: TonePipe(1.02 * f, SR, level=0.5), 440.0, SR)
    assert design == pytest.approx(440.0 / 1.02, rel=1e-4)
    assert gain * 0.5 / math.sqrt(2) == pytest.approx(0.08, rel=0.01)  # reference level at 440 Hz
    # Basses are regulated fuller than trebles.
    assert voice(lambda f: TonePipe(f, SR), 110.0, SR)[1] > gain


def test_voice_rejects_a_pipe_that_does_not_speak():
    with pytest.raises(PipeDoesNotSpeakError):
        voice(lambda f: TonePipe(f, SR, level=0.0), 440.0, SR)
    with pytest.raises(PipeDoesNotSpeakError):  # overblown into the octave
        voice(lambda f: TonePipe(2 * f, SR), 440.0, SR)


# --- keyboard and stops, with stand-in pipes ---------------------------------------


def test_a_key_sounds_one_pipe_per_stop(organ):
    out = organ.render(num_samples=4000, frequency=220.0, stops=(8, 4, 2))
    assert sorted(pipe.frequency for pipe in TonePipe.built) == pytest.approx([220, 440, 880])
    levels = harmonic_levels(out, SR, 220.0, 4)
    assert levels[0] > levels[1] > levels[3] > 10 * levels[2]  # 220, 440, 880 but no 660 Hz


def test_stop_length_sets_the_pitch(organ):
    out = organ.render(num_samples=4000, frequency=220.0, stops=[4])
    assert measure_tone(out, SR, 440.0)[0] == pytest.approx(440.0, abs=0.01)
    mutation = organ.render(num_samples=4000, frequency=220.0, stops=[8 / 3])  # a twelfth
    assert measure_tone(mutation[100:], SR, 660.0)[0] == pytest.approx(660.0, abs=0.01)


def test_same_key_is_held_across_calls(organ):
    pieces = np.concatenate([organ.render(num_samples=n) for n in (1, 299, 0, 600)])
    assert len(TonePipe.built) == 1  # one pipe, not one per call
    whole = ToneOrgan(sample_rate=SR, tuning_spread_cents=0.0).render(num_samples=900)
    np.testing.assert_allclose(pieces, whole, atol=1e-6)


def test_new_key_releases_the_old_one_which_rings_out(organ):
    organ.render(num_samples=1000, frequency=220.0)
    out = organ.render(num_samples=1000, frequency=330.0)
    alone = ToneOrgan(sample_rate=SR, tuning_spread_cents=0.0).render(
        num_samples=1000, frequency=330.0
    )
    assert np.max(np.abs((out - alone)[: TonePipe.FADE - 1])) > 1e-3  # old pipe still heard
    np.testing.assert_allclose(out[TonePipe.FADE :], alone[TonePipe.FADE :], atol=1e-6)


def test_rest_releases_the_key_and_silence_follows(organ):
    organ.render(num_samples=1000, frequency=220.0)
    tail = organ.render(num_samples=1000, amplitude=0.0)
    assert np.max(np.abs(tail[:10])) > 1e-3
    assert not tail[TonePipe.FADE :].any()
    assert not organ.render(num_samples=100, amplitude=0.0, frequency=5.0).any()


def test_repressing_a_ringing_key_reuses_its_pipes(organ):
    organ.render(num_samples=1000, frequency=220.0)
    organ.render(num_samples=10, amplitude=0.0)  # shorter than the fade
    again = organ.render(num_samples=1000, frequency=220.0)
    assert len(TonePipe.built) == 1
    assert rms(again[-500:]) == pytest.approx(0.08 * 2**0.25, rel=0.02)


def test_amplitude_scales_and_changes_smoothly(organ):
    loud = organ.render(num_samples=2000, frequency=220.0, amplitude=1.0)
    soft = organ.render(num_samples=2000, frequency=220.0, amplitude=0.5)
    assert rms(soft[-1000:]) == pytest.approx(0.5 * rms(loud[-1000:]), rel=0.01)
    steps = np.abs(np.diff(np.concatenate([loud, soft])))
    assert steps.max() < 1.05 * np.abs(np.diff(loud)).max()  # no click at the change


def test_each_pipe_has_its_own_fixed_mistuning():
    TonePipe.built = []
    organ = ToneOrgan(sample_rate=SR, tuning_spread_cents=2.0)
    organ.render(num_samples=10, frequency=220.0, stops=(8, 4))
    organ.render(num_samples=10, frequency=440.0, stops=(8,))
    low8, low4, high8 = (pipe.frequency for pipe in TonePipe.built)
    for sounding, nominal in ((low8, 220.0), (low4, 440.0), (high8, 440.0)):
        assert 0 < abs(cents(sounding, nominal)) <= 2.0
    assert low4 != high8  # same pitch, different ranks: different pipes
    # The same organ builds the same pipes again; another seed gives another organ.
    again = ToneOrgan(sample_rate=SR, tuning_spread_cents=2.0)
    again.render(num_samples=10, frequency=220.0, stops=(8, 4))
    assert [pipe.frequency for pipe in TonePipe.built[-2:]] == [low8, low4]
    other = ToneOrgan(sample_rate=SR, tuning_spread_cents=2.0, seed=1)
    other.render(num_samples=10, frequency=220.0, stops=(8, 4))
    assert [pipe.frequency for pipe in TonePipe.built[-2:]] != [low8, low4]


def test_reset_silences_and_restarts(organ):
    first = organ.render(num_samples=500, frequency=220.0)
    organ.render(num_samples=500, frequency=330.0)
    organ.reset()
    assert not organ.render(num_samples=100, amplitude=0.0).any()
    np.testing.assert_array_equal(organ.render(num_samples=500, frequency=220.0), first)


@pytest.mark.parametrize(
    "params",
    [
        {"amplitude": -0.1},
        {"amplitude": 1.1},
        {"frequency": 0.0},
        {"frequency": -440.0},
        {"frequency": SR / 8 + 1},
        {"frequency": 1000.0, "stops": (8, 2)},  # the 2' pipe would be too high
        {"stops": ()},
        {"stops": (8, 0)},
        {"stops": (-4,)},
    ],
)
def test_invalid_parameters(organ, params):
    with pytest.raises(ValueError):
        organ.render(num_samples=10, **params)


def test_invalid_constructor_arguments():
    with pytest.raises(ValueError):
        ToneOrgan(sample_rate=SR, tuning_spread_cents=-1.0)
    with pytest.raises(ValueError):
        ToneOrgan(sample_rate=0)


def test_unknown_parameter_rejected(organ):
    with pytest.raises(TypeError):
        organ.render(num_samples=10, cutoff=1000.0)


def test_pipes_too_high_for_the_sample_rate_are_oversampled():
    TonePipe.built = []

    class Fine(ToneOrgan):
        samples_per_period = 50.0

    organ = Fine(sample_rate=SR, tuning_spread_cents=0.0)
    assert [organ.oversampling(f) for f in (100.0, 441.0, 442.0, 882.0, 883.0, 2700.0)] == [
        1, 1, 2, 2, 3, 7,
    ]  # fmt: skip
    low = organ.render(num_samples=4000, frequency=220.0)
    organ.reset()
    high = organ.render(num_samples=4000, frequency=1000.0)
    assert [pipe.sample_rate for pipe in TonePipe.built] == [SR, 3 * SR]
    # Both come out at the synth's sample rate, tuned and regulated alike ...
    assert measure_tone(low[500:], SR, 220.0)[0] == pytest.approx(220.0, abs=0.01)
    assert measure_tone(high[500:], SR, 1000.0)[0] == pytest.approx(1000.0, abs=0.01)
    assert rms(high[2000:]) == pytest.approx(0.08 * (1000 / 440) ** -0.25, rel=0.02)
    # ... the oversampled one a millisecond (48 samples) late.
    assert np.abs(low[:30]).max() > 0.5 * np.abs(low).max()
    assert np.abs(high[:30]).max() < 0.02 * np.abs(high).max()
    assert np.abs(high[48:78]).max() > 0.5 * np.abs(high).max()
    # By default nothing is oversampled.
    assert ToneOrgan(sample_rate=SR).oversampling(2700.0) == 1


def test_tunings_are_remembered_until_forgotten():
    voiced = []

    class Counting(ToneOrgan):
        def _build_pipe(self, design_frequency, sample_rate, noise_seed):
            if noise_seed == 0:  # built for voicing
                voiced.append(design_frequency)
            return super()._build_pipe(design_frequency, sample_rate, noise_seed)

    Counting(sample_rate=SR).render(num_samples=10, frequency=220.0)
    Counting(sample_rate=SR).render(num_samples=10, frequency=220.0)
    assert len(voiced) == 1  # the second organ takes over the tuning of the first
    Counting(sample_rate=SR).render(num_samples=10, frequency=330.0)
    Counting(sample_rate=2 * SR).render(num_samples=10, frequency=220.0)
    Counting(sample_rate=SR, voicing=VOICING).render(num_samples=10, frequency=220.0)
    assert len(voiced) == 4  # another pitch, sample rate or voicing is another pipe
    PipeOrganSynth.forget_tunings()
    Counting(sample_rate=SR).render(num_samples=10, frequency=220.0)
    assert len(voiced) == 5


# --- the real organs ---------------------------------------------------------------


@pytest.fixture(params=ORGANS)
def real_organ(request):
    return create_synth(request.param, sample_rate=SR, tuning_spread_cents=0.0)


def test_output_format(real_organ):
    out = real_organ.render(duration=0.1, frequency=220.0, stops=(8, 4, 2))
    assert out.shape == (2205,) and out.dtype == np.float32
    assert np.isfinite(out).all() and np.max(np.abs(out)) < 1.0


@pytest.mark.parametrize("frequency", [55.0, 130.81, 440.0, 1046.5, 2637.0])
def test_pipes_are_in_tune_and_regulated(real_organ, frequency):
    periods = SR / frequency
    out = real_organ.render(num_samples=round(150 * periods), frequency=frequency)
    steady = out[round(90 * periods) :].astype(np.float64)
    sounding, fundamental = measure_tone(steady, SR, frequency)
    assert abs(cents(sounding, frequency)) < 1.0
    assert rms(steady) == pytest.approx(0.08 * (frequency / 440.0) ** -0.25, rel=0.05)
    assert fundamental > rms(steady)  # the fundamental carries the tone


@pytest.mark.parametrize("frequency", [110.0, 440.0])
def test_tone_of_a_principal(real_organ, frequency):
    periods = SR / frequency
    out = real_organ.render(num_samples=round(140 * periods), frequency=frequency)
    steady = out[round(100 * periods) :].astype(np.float64)
    levels = 20 * np.log10(harmonic_levels(steady, SR, frequency, 8))
    levels -= levels[0]
    assert -12 < levels[1] < 0  # a strong octave: the pipe is open at both ends
    assert -30 < levels[2] < -3
    assert np.all(levels[1:] < 0)
    assert levels[7] < -20  # but no buzz
    # The tone is steady: no vibrato, no beating.
    half = len(steady) // 2
    assert rms(steady[:half]) == pytest.approx(rms(steady[half:]), rel=0.02)


def test_bass_pipes_are_richer_in_harmonics_than_trebles(real_organ):
    def brightness(frequency):
        periods = SR / frequency
        out = real_organ.render(num_samples=round(140 * periods), frequency=frequency)
        levels = harmonic_levels(out[round(100 * periods) :].astype(np.float64), SR, frequency, 4)
        return sum(levels[2:]) / levels[0]

    assert brightness(82.4) > brightness(1318.5)


def test_pipe_speaks_gradually_and_rings_out(real_organ):
    frequency = 220.0
    period = round(SR / frequency)
    out = real_organ.render(num_samples=120 * period, frequency=frequency)
    steady = rms(out[100 * period :])
    assert rms(out[: 2 * period]) < 0.1 * steady
    assert rms(out[4 * period : 6 * period]) < 0.5 * steady
    assert rms(out[45 * period : 50 * period]) > 0.9 * steady
    assert np.max(np.abs(out)) < 1.5 * np.max(np.abs(out[100 * period :]))  # no blare

    tail = real_organ.render(num_samples=SR, amplitude=0.0)
    assert rms(tail[:period]) > 0.5 * steady  # the pipe does not stop dead
    assert rms(tail[30 * period : 32 * period]) < 0.05 * steady
    assert np.max(np.abs(tail[-100:])) < 1e-6
    assert not real_organ.render(num_samples=100, amplitude=0.0).any()  # silent in the end


def test_breath_noise_is_present_but_faint(real_organ):
    frequency = 440.0
    periods = SR / frequency
    out = real_organ.render(num_samples=round(200 * periods), frequency=frequency)
    steady = out[round(100 * periods) :].astype(np.float64)
    sounding, _ = measure_tone(steady, SR, frequency)
    spectrum = np.abs(np.fft.rfft(steady * np.hanning(len(steady)))) ** 2
    bins = np.arange(len(spectrum)) * SR / len(steady) / sounding
    between = (np.abs(bins - np.round(bins)) > 0.25) & (bins > 0.5)
    noise = spectrum[between].sum() / spectrum.sum()
    assert 1e-7 < noise < 1e-3


def test_rendering_in_blocks_is_seamless(real_organ):
    whole = real_organ.render(num_samples=3000, frequency=330.0, stops=(8, 4))
    real_organ.reset()
    pieces = np.concatenate(
        [real_organ.render(num_samples=n, frequency=330.0, stops=(8, 4)) for n in (1, 999, 0, 2000)]
    )
    np.testing.assert_allclose(pieces, whole, atol=1e-6)


def test_legato_overlaps_and_repeated_note_needs_a_rest(real_organ):
    period = round(SR / 220.0)
    real_organ.render(num_samples=80 * period, frequency=220.0)
    out = real_organ.render(num_samples=80 * period, frequency=330.0).astype(np.float64)
    # Right after the change the old pipe still dominates, later only the new one sounds.
    start, end = out[: 4 * period], out[-40 * period :]
    assert harmonic_levels(start, SR, 220.0, 1)[0] > harmonic_levels(start, SR, 330.0, 1)[0]
    assert harmonic_levels(end, SR, 330.0, 1)[0] > 100 * harmonic_levels(end, SR, 220.0, 1)[0]

    # Same key again without a rest: nothing happens. With a rest: it speaks anew.
    held = real_organ.render(num_samples=10 * period, frequency=330.0)
    assert rms(held) == pytest.approx(rms(end), rel=0.05)
    real_organ.render(num_samples=40 * period, amplitude=0.0)
    again = real_organ.render(num_samples=3 * period, frequency=330.0)
    assert rms(again) < 0.5 * rms(end)


@pytest.mark.parametrize("name", ORGANS)
def test_ranks_of_a_chorus_are_slightly_out_of_tune(name):
    organ = create_synth(name, sample_rate=SR)
    periods = SR / 880.0
    out = organ.render(num_samples=round(150 * periods), frequency=440.0, stops=(4,))
    sounding, _ = measure_tone(out[round(100 * periods) :].astype(np.float64), SR, 880.0)
    assert 0.01 < abs(cents(sounding, 880.0)) < 1.6


def test_reset_reproduces_the_same_sound(real_organ):
    first = real_organ.render(num_samples=2000, frequency=262.0)
    real_organ.render(num_samples=500, frequency=392.0)
    real_organ.reset()
    np.testing.assert_array_equal(real_organ.render(num_samples=2000, frequency=262.0), first)


@pytest.mark.parametrize("name", ORGANS)
def test_voicing_can_be_changed(name):
    def steady(**kwargs):
        organ = create_synth(name, sample_rate=SR, tuning_spread_cents=0.0, **kwargs)
        assert organ.voicing == kwargs.get("voicing", PRINCIPAL)
        periods = SR / 220.0
        out = organ.render(num_samples=round(170 * periods), frequency=220.0)
        return out[round(100 * periods) :].astype(np.float64)

    def between_harmonics(tone):
        sounding, _ = measure_tone(tone, SR, 220.0)
        spectrum = np.abs(np.fft.rfft(tone * np.hanning(len(tone)))) ** 2
        bins = np.arange(len(spectrum)) * SR / len(tone) / sounding
        return spectrum[np.abs(bins - np.round(bins)) > 0.25].sum() / spectrum.sum()

    usual = steady()
    # Without turbulence in the jet there is no breath noise.
    silent = steady(voicing=dataclasses.replace(PRINCIPAL, turbulence=0.0))
    assert between_harmonics(silent) < 0.01 * between_harmonics(usual)
    # A jet that swings further saturates more: a brighter tone, still tuned and regulated.
    bright = steady(voicing=dataclasses.replace(PRINCIPAL, jet_swing=1.5))
    assert abs(cents(measure_tone(bright, SR, 220.0)[0], 220.0)) < 1.0
    assert rms(bright) == pytest.approx(rms(usual), rel=0.05)
    change = np.divide(harmonic_levels(bright, SR, 220.0, 6), harmonic_levels(usual, SR, 220.0, 6))
    assert np.abs(20 * np.log10(change / change[0])).max() > 2


@pytest.mark.parametrize("name", ORGANS)
def test_high_pipes_of_the_real_organs_are_oversampled(name):
    organ = create_synth(name, sample_rate=SR, tuning_spread_cents=0.0)
    assert organ.samples_per_period >= 50
    assert organ.oversampling(110.0) == 1 < organ.oversampling(880.0) < organ.oversampling(2637.0)
    # At twice the sample rate half the factor will do.
    fine = create_synth(name, sample_rate=2 * SR)
    assert fine.oversampling(2637.0) == math.ceil(organ.oversampling(2637.0) / 2)
    # An oversampled pipe speaks a millisecond late and is otherwise a pipe like any other.
    out = organ.render(num_samples=6000, frequency=1760.0)
    assert np.abs(out[:20]).max() < 1e-4 * np.abs(out).max()
    assert rms(out[3000:]) == pytest.approx(0.08 * (1760 / 440) ** -0.25, rel=0.05)
    tail = organ.render(num_samples=SR, amplitude=0.0)
    assert np.abs(tail[-100:]).max() < 1e-6
    assert not organ.render(num_samples=100, amplitude=0.0).any()
