"""Listening to synth output: play it on the sound card or keep it in WAV files."""

import os
import wave
from collections.abc import Iterable, Mapping

import numpy as np

from synthplayground.synth import Synth


def play(
    synth: Synth,
    segments: Iterable[Mapping],
    filename: str | os.PathLike | None = None,
) -> np.ndarray:
    """Render consecutive segments with ``synth`` and play or save the result.

    Each dict in ``segments`` is passed as keyword arguments to one
    ``synth.render()`` call, so it must contain ``num_samples`` or ``duration``
    plus any sound parameters, e.g. ``{"duration": 0.5, "frequency": 440.0}``.
    The rendered segments are concatenated. If ``filename`` is given the audio
    is written there as a 16-bit mono WAV file; otherwise it is played with
    sounddevice, blocking until playback has finished.

    Returns the concatenated audio.
    """
    audio = render_segments(synth, segments)
    if filename is not None:
        write_wav(filename, audio, synth.sample_rate)
    else:
        # Imported lazily: sounddevice is optional and needs PortAudio.
        import sounddevice as sd

        sd.play(audio, synth.sample_rate)
        sd.wait()
    return audio


def render_segments(synth: Synth, segments: Iterable[Mapping]) -> np.ndarray:
    """Render consecutive segments with ``synth`` and return them concatenated.

    Each dict in ``segments`` is passed as keyword arguments to one
    ``synth.render()`` call (see :func:`play`).
    """
    segments = list(segments)
    if not segments:
        raise ValueError("segments must contain at least one dict of render parameters")
    return np.concatenate([synth.render(**params) for params in segments])


def mix(tracks: Iterable[np.ndarray]) -> np.ndarray:
    """Sum mono tracks into one; shorter tracks are padded with silence at the end.

    This is how several voices are played together: render each voice with
    its own synth instance (synths are monophonic and stateful) and mix the
    results. The sum is not scaled, so it may exceed [-1, 1]; see
    :func:`normalize`.
    """
    tracks = [np.asarray(track) for track in tracks]
    if not tracks:
        raise ValueError("tracks must contain at least one array")
    if any(track.ndim != 1 for track in tracks):
        raise ValueError("tracks must be mono (1-D) arrays")
    out = np.zeros(max(len(track) for track in tracks), dtype=np.float32)
    for track in tracks:
        out[: len(track)] += track
    return out


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale ``audio`` so that its largest absolute value is ``peak``; silence stays silent."""
    if not 0 < peak <= 1:
        raise ValueError(f"peak must be in (0, 1], got {peak}")
    audio = np.asarray(audio, dtype=np.float32)
    largest = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio * np.float32(peak / largest) if largest > 0 else audio


def write_wav(filename: str | os.PathLike, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono audio in [-1, 1] as a 16-bit PCM WAV file (values outside are clipped)."""
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(os.fspath(filename), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def read_wav(filename: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Read a 16-bit mono PCM WAV file, as written by :func:`write_wav`.

    Returns the audio as float32 in [-1, 1] and its sample rate.
    """
    with wave.open(os.fspath(filename), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError(
                f"{filename} is not 16-bit mono "
                f"({f.getnchannels()} channels, {8 * f.getsampwidth()} bits)"
            )
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        return (pcm / 32767).astype(np.float32), f.getframerate()
