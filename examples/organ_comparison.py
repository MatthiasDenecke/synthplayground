"""Compare the two physically modelled pipe organs on the same piece of music.

Renders the opening of J. S. Bach's Toccata in D minor (BWV 565) on a
principal chorus (8' + 4' + 2') with each organ synth, once dry and once in a
church, and writes the results as WAV files::

    python examples/organ_comparison.py            # -> output/organ_*.wav
    python examples/organ_comparison.py some/dir

The organ synths are monophonic like every synth here, so the piece is
written as several voices; each voice is rendered by its own synth instance
(in its own process, to save time) and the voices are mixed.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from synthplayground import (
    church_reverb,
    create_synth,
    mix,
    normalize,
    render_segments,
    write_wav,
)

SAMPLE_RATE = 44100
SYNTHS = ("organ_waveguide", "organ_modal")
STOPS = (8, 4, 2)  # principal chorus

_SEMITONES = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}


def pitch(name: str) -> float:
    """Frequency of a note such as ``"A4"``, ``"C#5"`` or ``"Bb3"`` (A4 = 440 Hz)."""
    semitone = _SEMITONES[name[0]] + name.count("#") - name.count("b")
    octave = int(name.lstrip("ABCDEFG#b"))
    return 440.0 * 2 ** (semitone / 12 + octave - 4)


def transpose(name: str, octaves: int) -> str:
    return f"{name[:-1]}{int(name[-1]) + octaves}"


def toccata() -> list[list[tuple[float, float, str]]]:
    """The score: one list of ``(start, end, note)`` per voice, times in seconds.

    Three flourishes, each an octave below the last and played in octaves,
    then a diminished seventh chord piled up note by note over a pedal D and
    resolved into D major. The rhythm is free, as the fermatas of the
    original ask for; the spacing of the final chord is an arrangement.
    """
    upper: list[tuple[float, float, str]] = []
    now = 0.2

    def phrase(notes: list[tuple[str, float]]) -> None:
        nonlocal now
        for name, duration in notes:
            if name != "rest":
                upper.append((now, now + duration, name))
            now += duration

    phrase([("A5", 0.14), ("G5", 0.14), ("A5", 1.8), ("rest", 0.5)])
    phrase([("G5", 0.12), ("F5", 0.12), ("E5", 0.12), ("D5", 0.12), ("C#5", 0.45), ("D5", 1.9)])
    phrase([("rest", 0.8)])
    phrase([("A4", 0.15), ("G4", 0.15), ("A4", 1.8), ("rest", 0.5)])
    phrase([("E4", 0.16), ("F4", 0.16), ("C#4", 0.5), ("D4", 1.9), ("rest", 0.8)])
    phrase([("A3", 0.16), ("G3", 0.16), ("A3", 1.8), ("rest", 0.5)])
    phrase([("G3", 0.14), ("F3", 0.14), ("E3", 0.14), ("D3", 0.14), ("C#3", 0.5), ("D3", 2.0)])
    phrase([("rest", 0.9)])
    lower = [(start, end, transpose(name, -1)) for start, end, name in upper]

    # Pedal D, then the chord: each note enters in turn and is held.
    resolve = now + 5.4
    suspension = resolve + 0.9
    end = resolve + 4.9
    pedal = [(now, end, "D2")]
    chord = []
    resolutions = [
        ("C#3", "D3"),
        ("E3", None),  # this voice drops out
        ("G3", "F#3"),
        ("Bb3", "A3"),
        ("C#4", "D4"),
        ("E4", "F#4"),
    ]
    for i, (dissonance, consonance) in enumerate(resolutions):
        entry = now + 1.0 + 0.32 * i
        # The G hangs over as a suspension before it falls to the F sharp.
        change = suspension if dissonance == "G3" else resolve
        voice = [(entry, change, dissonance)]
        if consonance is not None:
            voice.append((change, end, consonance))
        chord.append(voice)

    # The hands are free again by the time the chord starts.
    return [upper + chord[5], lower + chord[4], chord[3], chord[2], chord[1], chord[0], pedal]


def segments(voice: list[tuple[float, float, str]], end: float) -> list[dict]:
    """Turn the notes of one voice into ``render()`` parameters, rests included."""
    out = []
    position = 0  # in samples, so that all voices stay aligned
    for start, stop, name in [*voice, (end, end, "rest")]:
        first, last = round(start * SAMPLE_RATE), round(stop * SAMPLE_RATE)
        if first > position:
            out.append({"num_samples": first - position, "amplitude": 0.0, "stops": STOPS})
        if last > first:
            out.append({"num_samples": last - first, "frequency": pitch(name), "stops": STOPS})
        position = max(position, last)
    return out


def render_voice(job: tuple[str, list[dict]]) -> np.ndarray:
    synth_name, voice_segments = job
    return render_segments(create_synth(synth_name, sample_rate=SAMPLE_RATE), voice_segments)


def render(synth_name: str) -> np.ndarray:
    voices = toccata()
    # Leave time for the last chord to ring out.
    end = max(stop for voice in voices for _, stop, _ in voice) + 1.0
    jobs = [(synth_name, segments(voice, end)) for voice in voices]
    with ProcessPoolExecutor() as pool:  # the voices are independent: one process each
        return mix(pool.map(render_voice, jobs))


def main(directory: str = "output") -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for name in SYNTHS:
        started = time.perf_counter()
        dry = render(name)
        versions = {"dry": dry, "church": church_reverb(dry, SAMPLE_RATE)}
        for version, audio in versions.items():
            path = out / f"{name}_{version}.wav"
            write_wav(path, normalize(audio), SAMPLE_RATE)
            print(f"{path}  ({len(audio) / SAMPLE_RATE:.1f} s)")
        print(f"  {name} rendered in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main(*sys.argv[1:2])
