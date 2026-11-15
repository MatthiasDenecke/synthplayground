"""What does a synth cost to run?"""

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from synthplayground.audio import render_segments
from synthplayground.synth import Synth


@dataclass(frozen=True)
class Cost:
    """Time a synth takes to render some audio."""

    audio: float  # seconds of audio rendered
    first: float  # seconds the first rendering took, including anything done once
    render: float  # seconds a rendering takes after that (the fastest of the repeats)

    @property
    def realtime_factor(self) -> float:
        """Seconds of audio per second of computing; above 1 is faster than real time."""
        return self.audio / self.render

    @property
    def preparation(self) -> float:
        """Seconds spent once only, e.g. on voicing the pipes of an organ."""
        return max(0.0, self.first - self.render)


def measure_cost(
    make_synth: Callable[[], Synth], segments: Iterable[Mapping], repeats: int = 3
) -> Cost:
    """Time how long a synth takes to render ``segments``.

    ``make_synth()`` builds a fresh synth for each rendering, so that every
    rendering starts from silence; ``segments`` are as for
    :func:`synthplayground.audio.render_segments`. The first rendering is
    timed separately because synths may prepare things on first use and keep
    them (the organ synths tune and regulate each pipe once per process; call
    ``PipeOrganSynth.forget_tunings()`` beforehand to include that). After it
    come ``repeats`` more, of which the fastest counts, as the least disturbed
    by whatever else the computer is doing.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    segments = list(segments)
    times = []
    for _ in range(repeats + 1):
        synth = make_synth()
        start = time.perf_counter()
        audio = render_segments(synth, segments)
        times.append(time.perf_counter() - start)
    return Cost(audio=len(audio) / synth.sample_rate, first=times[0], render=min(times[1:]))
