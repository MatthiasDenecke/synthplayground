"""Listening to synth output: play it on the sound card or write it to a WAV file."""

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
    segments = list(segments)
    if not segments:
        raise ValueError("segments must contain at least one dict of render parameters")
    audio = np.concatenate([synth.render(**params) for params in segments])
    if filename is not None:
        write_wav(filename, audio, synth.sample_rate)
    else:
        # Imported lazily: sounddevice is optional and needs PortAudio.
        import sounddevice as sd

        sd.play(audio, synth.sample_rate)
        sd.wait()
    return audio


def write_wav(filename: str | os.PathLike, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono audio in [-1, 1] as a 16-bit PCM WAV file (values outside are clipped)."""
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(os.fspath(filename), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
