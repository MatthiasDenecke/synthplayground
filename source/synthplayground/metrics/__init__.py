"""Metrics for assessing the output of synths without reference recordings.

* :mod:`~synthplayground.metrics.tone`: measurements on notes and audio, and
  the difference between two synths,
* :mod:`~synthplayground.metrics.physics`: resonances of pipe models against
  theory, tuning, scaling over the compass,
* :mod:`~synthplayground.metrics.numerics`: aliasing, convergence with the
  sample rate, stability over all pitches,
* :mod:`~synthplayground.metrics.cost`: computing time.

None of these says how real a synth sounds; that takes recordings of the
real thing, or ears.
"""

from synthplayground.metrics.cost import Cost, measure_cost
from synthplayground.metrics.numerics import (
    Stability,
    aliasing_level,
    convergence,
    inharmonic_tones,
    stability,
)
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
from synthplayground.metrics.tone import (
    OCTAVE_BANDS,
    Note,
    NoteDifference,
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

__all__ = [
    "OCTAVE_BANDS",
    "Cost",
    "Note",
    "NoteDifference",
    "PowerLaw",
    "Resonance",
    "ResonanceDeviation",
    "Stability",
    "aliasing_level",
    "attack_time",
    "band_levels",
    "cents",
    "compass",
    "convergence",
    "detailed_resonances",
    "envelope",
    "ideal_resonances",
    "inharmonic_level",
    "inharmonic_peaks",
    "inharmonic_tones",
    "level_db",
    "measure_cost",
    "measure_note",
    "measure_notes",
    "measure_resonances",
    "note_difference",
    "partial_levels",
    "power_law",
    "release_time",
    "resonance_deviations",
    "sounding_pitch",
    "stability",
]
