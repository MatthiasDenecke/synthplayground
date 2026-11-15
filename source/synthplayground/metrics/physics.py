"""Does a pipe model behave like a pipe? Resonances against theory, and scaling laws.

The resonances of a simulated pipe are measured from its sound: the pipe is
tapped (:meth:`synthplayground.organ.FluePipe.strike`) with the wind off and
rings at its own resonances, whose frequencies and decay times are then read
from the ringing. They are compared with two references:

* the *detailed* theory (:func:`detailed_resonances`): end corrections that
  shrink with frequency, losses to the walls, to radiation and at the mouth.
  The modal synth is built from these very formulas, so it can deviate only
  through its discretisation; the waveguide approximates them with filters.
* the *ideal* textbook pipe (:func:`ideal_resonances`): a cylinder open at
  both ends without a mouth, whose resonances are exact harmonics. Part of
  what a model deviates from this is physics, not error.

Tuning errors are in :attr:`synthplayground.metrics.tone.Note.detuning`;
:func:`power_law` checks how a measurement scales over the compass.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice

import numpy as np

from synthplayground.metrics.tone import cents
from synthplayground.organ import (
    PRINCIPAL,
    SPEED_OF_SOUND,
    FluePipe,
    PipeGeometry,
    Voicing,
    pipe_modes,
)


_BACKGROUND = 2  # degree of the polynomial that stands for the tails of the other resonances


@dataclass(frozen=True)
class Resonance:
    """A resonance of a pipe."""

    number: int  # 1 is the fundamental
    frequency: float  # Hz
    decay_time: float  # s in which it decays to 1/e when left alone


def detailed_resonances(
    frequency: float, count: int, voicing: Voicing = PRINCIPAL
) -> list[Resonance]:
    """Up to ``count`` resonances of the principal pipe for ``frequency`` by the detailed theory.

    See :func:`synthplayground.organ.pipe_modes`. Fewer are returned where the
    pipe gets too wide, compared to the wavelength, for the theory to hold.
    """
    modes = islice(pipe_modes(PipeGeometry.principal(frequency), voicing), count)
    return [Resonance(mode.number, mode.frequency, mode.decay_time) for mode in modes]


def ideal_resonances(frequency: float, count: int) -> list[Resonance]:
    """Up to ``count`` resonances of the textbook pipe: an ideal cylinder open at both ends.

    Its end corrections do not depend on frequency, so the resonances are the
    harmonics of ``frequency``. It loses sound to its walls and by radiation
    from the two open ends, each ``(ka)^2 / 2`` of the amplitude per
    reflection; there is no mouth. That radiation formula is for low
    frequencies, so the series stops at ``ka = 1``.
    """
    geometry = PipeGeometry.principal(frequency)
    resonances = []
    for number in range(1, count + 1):
        ka = 2 * math.pi * number * frequency * geometry.radius / SPEED_OF_SOUND
        if ka >= 1:
            break
        loss = geometry.wall_loss(number * frequency) + ka * ka  # per round trip
        resonances.append(Resonance(number, number * frequency, 1 / (frequency * loss)))
    return resonances


def measure_resonances(
    pipe: FluePipe, sample_rate: int, frequency: float, count: int = 8, amount: float = 1e-4
) -> list[Resonance]:
    """Up to ``count`` resonances of a simulated ``pipe``, read from how it rings when tapped.

    ``pipe`` must be fresh (never blown) and built for ``frequency``, which
    tells where to look for the first resonance; each further one is looked
    for about as far above the last as that was above the one before. The
    search ends early where the resonances are too damped to show as peaks.
    ``amount`` is the strength of the tap.

    Each resonance is found by fitting a single pole plus a smooth background
    to the spectrum of the ringing around its peak, which gives frequency and
    decay time free of the bias that picking peaks would have.
    """
    pipe.strike(amount)
    blocks = [pipe.render(4096, blowing=False)]
    loudest = np.max(np.abs(blocks[0]))
    if loudest == 0:
        raise ValueError("the pipe does not ring when struck")
    while np.max(np.abs(blocks[-1])) > 1e-7 * loudest and len(blocks) < 20 * sample_rate / 4096:
        blocks.append(pipe.render(4096, blowing=False))
        loudest = max(loudest, np.max(np.abs(blocks[-1])))
    ringing = np.concatenate(blocks)
    ringing = ringing[np.flatnonzero(ringing)[0] :]  # a delay would look like a phase slope

    size = 1 << (8 * len(ringing) - 1).bit_length()  # zero-padded: a finely sampled spectrum
    spectrum = np.fft.rfft(ringing, size)
    power = np.abs(spectrum) ** 2
    hertz = sample_rate / size  # per bin

    resonances: list[Resonance] = []
    last, spacing = 0.0, frequency
    for number in range(1, count + 1):
        centre = frequency if number == 1 else last + spacing
        low = math.ceil((centre - 0.35 * spacing) / hertz)
        high = min(math.floor((centre + 0.35 * spacing) / hertz), int(0.45 * size))
        if high - low < 8:
            break
        peak = low + int(np.argmax(power[low : high + 1]))
        if peak in (low, high):  # no peak inside: the resonances have merged into a slope
            break
        # Fit over the half-power width of the peak and as much again on either side.
        above = power >= 0.5 * power[peak]
        left = right = peak
        while left > low and above[left - 1]:
            left -= 1
        while right < high and above[right + 1]:
            right += 1
        reach = max(right - left, 4)
        bins = np.arange(max(left - reach, low), min(right + reach, high) + 1)

        # Near its peak the response is one pole plus the smooth tails of all others:
        #   H = c / (1 - p u) + background(u),  u = 1 / z turned so that u = 1 at the peak.
        # Times (1 - p u), with the background a polynomial in (u - 1), this is linear in p
        # and the coefficients: H = p (u H) + a0 + a1 (u - 1) + ... + a3 (u - 1)^3.
        response = spectrum[bins]
        turn = np.exp(-2j * np.pi * (bins - peak) / size)
        system = np.column_stack(
            [response * turn] + [(turn - 1) ** power for power in range(_BACKGROUND + 2)]
        )
        pole = np.linalg.lstsq(system, response, rcond=None)[0][0]
        pole *= np.exp(2j * np.pi * peak / size)
        if not 0 < abs(pole) < 1:
            break
        found = Resonance(
            number,
            frequency=float(np.angle(pole) * sample_rate / (2 * np.pi)),
            decay_time=float(-1 / (sample_rate * np.log(abs(pole)))),
        )
        if not low * hertz <= found.frequency <= high * hertz:
            break
        resonances.append(found)
        spacing = found.frequency - last
        last = found.frequency
    return resonances


@dataclass(frozen=True)
class ResonanceDeviation:
    """How a measured resonance differs from a reference."""

    number: int
    cents: float  # measured frequency above (positive) or below the reference
    decay_ratio: float  # measured decay time / reference decay time


def resonance_deviations(
    measured: Iterable[Resonance], reference: Iterable[Resonance]
) -> list[ResonanceDeviation]:
    """Deviation of each of the ``measured`` resonances that the ``reference`` also has."""
    expected = {resonance.number: resonance for resonance in reference}
    return [
        ResonanceDeviation(
            resonance.number,
            cents(resonance.frequency, expected[resonance.number].frequency),
            resonance.decay_time / expected[resonance.number].decay_time,
        )
        for resonance in measured
        if resonance.number in expected
    ]


@dataclass(frozen=True)
class PowerLaw:
    """A fit ``value = scale * frequency ** exponent``."""

    exponent: float
    scale: float
    worst: float  # largest deviation of a value from the fit, in dB (20 log10 of the ratio)

    def __call__(self, frequency: float) -> float:
        return self.scale * frequency**self.exponent


def power_law(frequencies: Sequence[float], values: Sequence[float]) -> PowerLaw:
    """Fit a power law to positive ``values`` measured at ``frequencies`` across a compass.

    The ranks of an organ are scaled: level, speech and timbre change
    smoothly and by a rule from the bass to the treble. The exponent tells
    the rule a model follows (e.g. -0.25 for the level the pipes are
    regulated to), ``worst`` how far a single pipe strays from it.
    """
    frequencies, values = np.asarray(frequencies, float), np.asarray(values, float)
    if frequencies.shape != values.shape or frequencies.ndim != 1 or len(frequencies) < 2:
        raise ValueError("need at least two frequencies and as many values")
    if not (np.all(frequencies > 0) and np.all(values > 0)):
        raise ValueError("frequencies and values must be positive")
    exponent, intercept = np.polyfit(np.log(frequencies), np.log(values), 1)
    residual = np.log(values) - (intercept + exponent * np.log(frequencies))
    return PowerLaw(
        float(exponent), float(np.exp(intercept)), float(20 / math.log(10) * np.abs(residual).max())
    )
