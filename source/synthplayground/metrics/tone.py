"""Measurements on the sound of a synth, and the differences between two synths.

The building blocks work on arrays of mono audio; :func:`measure_note` plays
one note on a synth and applies them, and :func:`note_difference` subtracts
two such measurements, e.g. of the same note on two synths.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from synthplayground.organ import measure_tone
from synthplayground.synth import Synth

#: Centre frequencies in Hz of the octave bands from 63 Hz to 16 kHz.
OCTAVE_BANDS = tuple(62.5 * 2.0**i for i in range(9))

_FRAME_PERIODS = 64  # length of an analysis frame, in periods of the tone
_GUARD_BINS = 8  # bins either side of a harmonic that belong to it (window main lobe)
_KAISER_BETA = 20.0  # side lobes below -150 dB
_TONE_FRAMES = 8  # frames needed to tell a tone from noise
_TONE_VARIATION = 0.5  # power of a tone varies less than this from frame to frame; noise: 1
_ENVELOPE_SAMPLES = 64  # shortest window for following the level of a tone


def cents(frequency: float, reference: float) -> float:
    """Interval from ``reference`` up to ``frequency`` in cents (1200 to the octave)."""
    return 1200 * math.log2(frequency / reference)


def level_db(ratio: float) -> float:
    """An amplitude ratio in decibels; ``-inf`` for zero."""
    return 20 * math.log10(ratio) if ratio > 0 else -math.inf


def sounding_pitch(tone: np.ndarray, sample_rate: int, nominal: float) -> float:
    """Fundamental frequency in Hz of a steady ``tone`` that is within a semitone of ``nominal``."""
    # A short stretch tolerates a nominal that is well off; the whole tone is then precise.
    rough, _ = measure_tone(tone[: round(24 * sample_rate / nominal)], sample_rate, nominal)
    return measure_tone(tone, sample_rate, rough)[0]


def partial_levels(tone: np.ndarray, sample_rate: int, pitch: float, count: int = 12) -> np.ndarray:
    """Levels in dB of the harmonics of a steady ``tone`` relative to the first.

    ``pitch`` is the fundamental frequency (see :func:`sounding_pitch`). Only
    harmonics below 0.45 times the sample rate are returned, at most ``count``.
    """
    count = min(count, int(0.45 * sample_rate / pitch))
    window = np.kaiser(len(tone), _KAISER_BETA)
    time = np.arange(len(tone))
    amplitudes = np.array(
        [
            abs(np.dot(tone, window * np.exp(-2j * np.pi * k * pitch / sample_rate * time)))
            for k in range(1, count + 1)
        ]
    )
    if count == 0 or amplitudes[0] == 0:
        raise ValueError("tone has no fundamental at the given pitch")
    with np.errstate(divide="ignore"):
        return 20 * np.log10(amplitudes / amplitudes[0])


def _harmonic_spectrum(
    tone: np.ndarray, sample_rate: int, pitch: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power spectrum of a steady tone and where its bins lie relative to the harmonics.

    The tone is cut into half-overlapping frames of ``_FRAME_PERIODS``
    periods. Returns the power in each bin for each frame (frames x bins),
    the position of each bin in units of the pitch, and a mask of the bins
    that lie between the harmonics.
    """
    frame = round(_FRAME_PERIODS * sample_rate / pitch)
    if len(tone) < frame:
        raise ValueError(f"tone must last at least {_FRAME_PERIODS} periods of its pitch")
    window = np.kaiser(frame, _KAISER_BETA)
    starts = range(0, len(tone) - frame + 1, frame // 2)
    power = np.array([np.abs(np.fft.rfft(tone[s : s + frame] * window)) ** 2 for s in starts])
    harmonic = np.arange(power.shape[1]) * sample_rate / frame / pitch
    between = np.abs(harmonic - np.round(harmonic)) * frame * pitch / sample_rate > _GUARD_BINS
    return power, harmonic, between


def inharmonic_level(tone: np.ndarray, sample_rate: int, pitch: float) -> float:
    """Share in dB of the power of a steady ``tone`` that is not in its harmonics.

    This is everything between the harmonics of ``pitch``: breath noise,
    aliasing, subharmonics, hum. The power found between the harmonics is
    scaled up for the bins the harmonics themselves occupy, assuming it is
    spread evenly. A perfectly periodic tone measures below -120 dB.
    """
    power, _, between = _harmonic_spectrum(tone, sample_rate, pitch)
    power = power.mean(axis=0)
    total = power.sum()
    if total == 0:
        raise ValueError("tone is silent")
    share = power[between].sum() / between.mean() / total
    return 10 * math.log10(share) if share > 0 else -math.inf


def inharmonic_peaks(
    tone: np.ndarray, sample_rate: int, pitch: float, margin: float = 10.0, floor: float = -120.0
) -> list[tuple[float, float]]:
    """Tones in a steady ``tone`` that do not belong to its harmonics.

    Looks between each pair of neighbouring harmonics of ``pitch`` for peaks
    that stand out by more than ``margin`` dB from what else is there (the
    median, i.e. the noise), are above ``floor`` dB, and are tones rather
    than noise: a tone has the same power in every stretch of the signal,
    whereas noise, even where a resonance gathers it into a narrow peak,
    comes and goes. To tell, ``tone`` must last at least 300 periods; the
    longer it is, the surer the verdict. Returns ``(frequency in Hz, level in
    dB relative to the fundamental)``, strongest first. Aliased partials show
    up here once they rise out of the noise.
    """
    frames, harmonic, between = _harmonic_spectrum(tone, sample_rate, pitch)
    if len(frames) < _TONE_FRAMES:
        raise ValueError("tone must last at least 300 periods to tell tones from noise")
    power = frames.mean(axis=0)
    fundamental = power[np.abs(harmonic - 1).argmin()]
    if fundamental == 0:
        raise ValueError("tone has no fundamental at the given pitch")
    peaks = []
    lower = np.floor(harmonic).astype(int)  # bins between harmonics n and n + 1 share n
    for n in range(lower[-1] + 1):
        bins = np.flatnonzero(between & (lower == n))
        if len(bins) < 3:
            continue
        noise = np.median(power[bins])
        threshold = max(noise * 10 ** (margin / 10), fundamental * 10 ** (floor / 10))
        for b in bins[1:-1]:
            if power[b] > threshold and power[b] >= power[b - 1] and power[b] > power[b + 1]:
                if frames[:, b].std() < _TONE_VARIATION * power[b]:
                    # The peak of the window is a parabola in dB: place the tone within the bin.
                    left, here, right = np.log(power[b - 1 : b + 2])
                    offset = 0.5 * (left - right) / (left - 2 * here + right)
                    frequency = (harmonic[b] + offset * (harmonic[1] - harmonic[0])) * pitch
                    peaks.append((float(frequency), 10 * math.log10(power[b] / fundamental)))
    return sorted(peaks, key=lambda peak: -peak[1])


def _envelope_periods(sample_rate: int, pitch: float) -> int:
    """Whole periods of ``pitch`` that span at least ``_ENVELOPE_SAMPLES`` samples."""
    return max(1, math.ceil(_ENVELOPE_SAMPLES * pitch / sample_rate))


def envelope(
    audio: np.ndarray, sample_rate: int, pitch: float, periods: float | None = None
) -> np.ndarray:
    """RMS level of ``audio`` in a sliding window of ``periods`` periods of ``pitch``.

    Element ``i`` is the level over the window that ends with sample ``i``.
    The window is as long as the periods to a fraction of a sample, so that
    the level of a steady tone comes out steady. By default it is one period,
    or as many whole periods as it takes to span 64 samples: the level of a
    tone with only a few samples to the period cannot be followed any closer.
    """
    if periods is None:
        periods = _envelope_periods(sample_rate, pitch)
    width = periods * sample_rate / pitch
    energy = np.concatenate([[0.0], np.cumsum(np.square(audio, dtype=np.float64))])
    ends = np.arange(1, len(audio) + 1)
    before = np.interp(np.maximum(ends - width, 0.0), np.arange(len(energy)), energy)
    return np.sqrt((energy[ends] - before) / width)


def attack_time(
    audio: np.ndarray, sample_rate: int, pitch: float, steady: float, share: float = 0.9
) -> float:
    """Seconds until a note first reaches ``share`` of its ``steady`` level.

    ``audio`` starts where the key is pressed. ``nan`` if it never gets there.
    """
    reached = np.flatnonzero(envelope(audio, sample_rate, pitch) >= share * steady)
    return float(reached[0] / sample_rate) if len(reached) else math.nan


def release_time(
    audio: np.ndarray, sample_rate: int, pitch: float, steady: float, drop: float = 20.0
) -> float:
    """Seconds until a released note falls ``drop`` dB below its ``steady`` level.

    ``audio`` starts where the key is released. ``nan`` if it never falls that far.
    """
    fallen = np.flatnonzero(envelope(audio, sample_rate, pitch) < steady * 10 ** (-drop / 20))
    # Until the window is full the level reads low for lack of samples, not of sound.
    window = math.ceil(_envelope_periods(sample_rate, pitch) * sample_rate / pitch)
    fallen = fallen[fallen >= window - 1]
    return float(fallen[0] / sample_rate) if len(fallen) else math.nan


def band_levels(
    audio: np.ndarray, sample_rate: int, centres: Iterable[float] = OCTAVE_BANDS
) -> np.ndarray:
    """Level in dB (relative to full scale) of ``audio`` in the octave bands around ``centres``.

    Subtract the results for two renderings of the same music to see where
    one is brighter or duller than the other. Bands with no power are ``-inf``.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("audio must be a non-empty mono (1-D) array")
    power = np.abs(np.fft.rfft(audio)) ** 2 * 2 / len(audio) ** 2  # mean square per bin
    frequency = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    levels = []
    for centre in centres:
        inside = (frequency >= centre / math.sqrt(2)) & (frequency < centre * math.sqrt(2))
        band = power[inside].sum()
        levels.append(10 * math.log10(band) if band > 0 else -math.inf)
    return np.array(levels)


@dataclass(frozen=True)
class Note:
    """What :func:`measure_note` finds out about one note of a synth."""

    nominal: float  # Hz, the pitch that was asked for
    pitch: float  # Hz, the pitch that sounds
    level: float  # RMS of the steady tone
    peak: float  # largest sample while the key is held
    partials: tuple[float, ...]  # dB relative to the first, which is therefore 0
    inharmonic: float  # dB, share of the power between the harmonics (noise, aliasing)
    fluctuation: float  # (largest - smallest) / mean of the level of the steady tone
    attack: float  # s from pressing the key until 90 % of the steady level
    overshoot: float  # highest level on the way, relative to the steady level
    release: float  # s from releasing the key until 20 dB below the steady level

    @property
    def detuning(self) -> float:
        """Cents by which the note sounds sharp (positive) or flat of what was asked for."""
        return cents(self.pitch, self.nominal)


def measure_note(
    synth: Synth,
    frequency: float,
    *,
    pitch: float | None = None,
    periods: int = 160,
    partials: int = 12,
    **params,
) -> Note:
    """Play one note on ``synth`` from silence, release it, and measure it.

    The synth is reset first. The key is held for ``periods`` periods, of
    which the last 72 are taken to be the steady tone, and then released
    (``amplitude=0``) for half as long. ``params`` are further sound
    parameters of the synth, e.g. ``stops=(8, 4, 2)``. ``pitch`` is the
    fundamental to expect if it is not ``frequency`` (a 4' stop alone sounds
    at twice the frequency of the key).
    """
    nominal = frequency if pitch is None else pitch
    period = synth.sample_rate / nominal
    if periods < 100:
        raise ValueError(f"periods must be at least 100, got {periods}")
    synth.reset()
    held = synth.render(num_samples=round(periods * period), frequency=frequency, **params)
    tail = synth.render(
        num_samples=round(periods / 2 * period), frequency=frequency, **{**params, "amplitude": 0.0}
    )
    held, tail = held.astype(np.float64), tail.astype(np.float64)
    if not (np.isfinite(held).all() and np.isfinite(tail).all()):
        raise FloatingPointError("the synth produced samples that are not finite")

    steady = held[-round(72 * period) :]
    level = float(np.sqrt(np.mean(steady**2)))
    if level == 0:
        raise ValueError("the note is silent")
    sounding = sounding_pitch(steady, synth.sample_rate, nominal)
    # Over 8 periods at a time: long enough to be exact, short enough to catch a warble.
    steadiness = envelope(steady, synth.sample_rate, sounding, periods=8)[round(8 * period) :]
    return Note(
        nominal=nominal,
        pitch=sounding,
        level=level,
        peak=float(np.max(np.abs(held))),
        partials=tuple(partial_levels(steady, synth.sample_rate, sounding, partials).tolist()),
        inharmonic=inharmonic_level(steady, synth.sample_rate, sounding),
        fluctuation=float((steadiness.max() - steadiness.min()) / steadiness.mean()),
        attack=attack_time(held, synth.sample_rate, nominal, level),
        overshoot=float(envelope(held, synth.sample_rate, nominal).max() / level),
        release=release_time(tail, synth.sample_rate, nominal, level),
    )


def measure_notes(synth: Synth, frequencies: Iterable[float], **params) -> list[Note]:
    """:func:`measure_note` for each of ``frequencies``, e.g. a :func:`compass`."""
    return [measure_note(synth, frequency, **params) for frequency in frequencies]


def compass(lowest: float, highest: float) -> list[float]:
    """Frequencies of the keys from ``lowest`` up to at most ``highest``, a semitone apart."""
    if not 0 < lowest <= highest:
        raise ValueError(f"need 0 < lowest <= highest, got {lowest} and {highest}")
    count = math.floor(12 * math.log2(highest / lowest) + 1e-9) + 1
    return [lowest * 2 ** (key / 12) for key in range(count)]


@dataclass(frozen=True)
class NoteDifference:
    """How one :class:`Note` differs from another (first minus second)."""

    pitch: float  # cents
    level: float  # dB
    partials: tuple[float, ...]  # dB, for the partials both notes have
    inharmonic: float  # dB
    attack: float  # s
    overshoot: float  # difference of the ratios
    release: float  # s


def note_difference(note: Note, reference: Note) -> NoteDifference:
    """``note`` minus ``reference``: the same note on two synths, or at two sample rates."""
    return NoteDifference(
        pitch=cents(note.pitch, reference.pitch),
        level=level_db(note.level / reference.level),
        partials=tuple(a - b for a, b in zip(note.partials, reference.partials)),
        inharmonic=note.inharmonic - reference.inharmonic,
        attack=note.attack - reference.attack,
        overshoot=note.overshoot - reference.overshoot,
        release=note.release - reference.release,
    )
