"""Audio effects applied to rendered audio (as opposed to synth algorithms)."""

import numpy as np

#: Reverberation time relative to ``rt60`` in octave bands from 63 Hz to 8 kHz.
#: Stone churches ring longest in the low-mid range; the air swallows the treble.
_CHURCH_DECAY = (0.95, 1.05, 1.1, 1.0, 0.85, 0.65, 0.45, 0.3)
_BAND_CENTRES = tuple(62.5 * 2**i for i in range(len(_CHURCH_DECAY)))


def church_impulse_response(
    sample_rate: int,
    rt60: float = 4.5,
    predelay: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic impulse response of the reverberation of a large stone church.

    The reverberant sound of a big room is, to the ear, noise that decays
    exponentially, more quickly at high frequencies. It is built that way:
    white noise is split into octave bands, each band decays with its own
    reverberation time (``rt60`` seconds at 500 Hz, see ``_CHURCH_DECAY``), and
    the sum fades in over the first few tens of milliseconds, ``predelay``
    seconds after the direct sound, as reflections arrive ever more densely.
    The direct sound itself is not included. The response has unit energy.
    """
    if rt60 <= 0:
        raise ValueError(f"rt60 must be positive, got {rt60}")
    if predelay < 0:
        raise ValueError(f"predelay must be non-negative, got {predelay}")
    length = int(1.2 * rt60 * max(_CHURCH_DECAY) * sample_rate)
    time = np.arange(length) / sample_rate
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(length))
    frequency = np.fft.rfftfreq(length, 1 / sample_rate)
    # Position of each frequency on the octave-band scale; bands cross-fade
    # with raised cosines, the outer bands extend to 0 Hz and Nyquist.
    position = np.log2(np.maximum(frequency, 1e-3) / _BAND_CENTRES[0])
    position = np.clip(position, 0, len(_BAND_CENTRES) - 1)
    response = np.zeros(length)
    for band, relative in enumerate(_CHURCH_DECAY):
        mask = np.cos(0.5 * np.pi * np.clip(np.abs(position - band), 0, 1)) ** 2
        decay = np.exp(-np.log(1000.0) * time / (rt60 * relative))
        response += np.fft.irfft(spectrum * mask, length) * decay
    response *= 1 - np.exp(-time / 0.02)  # build-up of the reflection density
    response = np.concatenate([np.zeros(round(predelay * sample_rate)), response])
    return response / np.sqrt(np.sum(response**2))


def church_reverb(
    audio: np.ndarray,
    sample_rate: int,
    rt60: float = 4.5,
    wet: float = 0.5,
    predelay: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    """Place mono ``audio`` in a large church.

    ``wet`` in [0, 1] is the share of reverberant sound: 0 returns the dry
    signal, 1 only the reverberation, as heard far down the nave. The result
    is longer than the input by the reverberation tail. Its level is not
    normalised; use :func:`synthplayground.audio.normalize` before writing it.
    """
    if not 0 <= wet <= 1:
        raise ValueError(f"wet must be in [0, 1], got {wet}")
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono (1-D), got shape {audio.shape}")
    response = church_impulse_response(sample_rate, rt60, predelay, seed)
    length = len(audio) + len(response) - 1
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    size = 1 << (length - 1).bit_length()
    reverberant = np.fft.irfft(np.fft.rfft(audio, size) * np.fft.rfft(response, size), size)
    out = wet * reverberant[:length]
    out[: len(audio)] += (1 - wet) * audio
    return out.astype(np.float32)
