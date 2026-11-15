"""Is the simulation numerically sound? Aliasing, convergence and stability.

* :func:`aliasing_level` and :func:`inharmonic_tones` look for sound that a
  periodic tone should not contain,
* :func:`convergence` renders the same note at several sample rates: a sound
  simulation gives the same answer when it is refined,
* :func:`stability` checks that every pitch of a compass speaks, holds
  steady and dies away.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from synthplayground.metrics.tone import (
    NoteDifference,
    inharmonic_peaks,
    measure_note,
    note_difference,
    sounding_pitch,
)
from synthplayground.synth import Synth


def aliasing_level(synth: Synth, frequency: float, **params) -> float:
    """Share in dB of the power of a steady note that lies between its harmonics.

    For a synth that makes no noise of its own (an organ synth built with
    ``voicing=dataclasses.replace(PRINCIPAL, turbulence=0)``) this is what the
    simulation adds by mistake: partials above half the sample rate folded
    back to where no partial belongs (aliasing), or a tone that does not
    repeat exactly. With breath noise switched on it measures mostly that
    noise; then use :func:`inharmonic_tones`. ``params`` as for
    :func:`synthplayground.metrics.tone.measure_note`.
    """
    return measure_note(synth, frequency, **params).inharmonic


def inharmonic_tones(
    synth: Synth,
    frequency: float,
    *,
    pitch: float | None = None,
    periods: int = 1000,
    margin: float = 10.0,
    **params,
) -> list[tuple[float, float]]:
    """Tones in a steady note of ``synth`` that stand out from the noise between its harmonics.

    This is the aliasing (or hum, or subharmonics) that can be told apart
    from the synth's own noise in normal operation. The key is held for
    ``periods`` periods; the first 100 are left out. Returns ``(frequency in
    Hz, level in dB relative to the fundamental)``, strongest first; see
    :func:`synthplayground.metrics.tone.inharmonic_peaks`.
    """
    nominal = frequency if pitch is None else pitch
    period = synth.sample_rate / nominal
    if periods < 400:
        raise ValueError(f"periods must be at least 400, got {periods}")
    synth.reset()
    held = synth.render(num_samples=round(periods * period), frequency=frequency, **params)
    steady = held[round(100 * period) :].astype(np.float64)
    sounding = sounding_pitch(steady, synth.sample_rate, nominal)
    return inharmonic_peaks(steady, synth.sample_rate, sounding, margin)


def convergence(
    make_synth: Callable[[int], Synth],
    frequency: float,
    sample_rates: Sequence[int] = (44100, 88200, 176400),
    **params,
) -> list[NoteDifference]:
    """How a note changes when the sample rate is raised.

    ``make_synth(sample_rate)`` builds the synth. The note is measured at each
    of ``sample_rates``; the result has, for every rate but the highest, its
    difference from the highest one, which stands for the exact solution.
    Small differences mean that the lower rate is fine enough. ``params`` as
    for :func:`synthplayground.metrics.tone.measure_note`.
    """
    if len(sample_rates) < 2 or max(sample_rates) != sample_rates[-1]:
        raise ValueError("need at least two sample rates, the highest last")
    notes = [measure_note(make_synth(rate), frequency, **params) for rate in sample_rates]
    return [note_difference(note, notes[-1]) for note in notes[:-1]]


@dataclass(frozen=True)
class Stability:
    """How one pitch of a synth fared in :func:`stability`."""

    frequency: float
    ok: bool
    problem: str  # empty if ok
    detuning: float  # cents; nan if the note could not be measured
    fluctuation: float  # of the level of the steady tone, see Note.fluctuation
    peak: float  # largest sample


def stability(
    synth: Synth,
    frequencies: Iterable[float],
    *,
    max_detuning: float = 50.0,
    max_fluctuation: float = 0.05,
    **params,
) -> list[Stability]:
    """Check that each of ``frequencies`` speaks, holds steady and dies away on ``synth``.

    A pitch fails if the synth refuses or cannot render it, if it sounds more
    than ``max_detuning`` cents off (a pipe that overblows speaks an octave or
    more too high), if the level of its steady tone varies by more than
    ``max_fluctuation`` (it warbles, beats or keeps growing), if it reaches
    full scale, or if it is not 20 dB down some 80 periods after the key is
    released. ``params`` as for
    :func:`synthplayground.metrics.tone.measure_note`.
    """
    results = []
    for frequency in frequencies:
        try:
            note = measure_note(synth, frequency, **params)
        except (ArithmeticError, RuntimeError, ValueError) as error:
            problem = f"{type(error).__name__}: {error}"
            results.append(Stability(frequency, False, problem, math.nan, math.nan, math.nan))
            continue
        problems = []
        if abs(note.detuning) > max_detuning:
            problems.append(f"sounds {note.detuning:+.0f} cents off")
        if note.fluctuation > max_fluctuation:
            problems.append(f"level varies by {note.fluctuation:.0%}")
        if note.peak >= 1.0:
            problems.append("reaches full scale")
        if math.isnan(note.release):
            problems.append("does not die away")
        summary = "; ".join(problems)
        results.append(
            Stability(frequency, not problems, summary, note.detuning, note.fluctuation, note.peak)
        )
    return results
