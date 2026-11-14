"""Abstract base class shared by all synth algorithms."""

from abc import ABC, abstractmethod

import numpy as np

DEFAULT_SAMPLE_RATE = 44100


class Synth(ABC):
    """Base class for a synth algorithm producing mono audio.

    Subclasses implement :meth:`_render`, declaring their sound parameters
    (e.g. frequency) as keyword arguments, which :meth:`render`
    passes through. Rendering may be stateful so that
    consecutive calls produce one continuous signal (e.g. an oscillator keeps
    its phase); call :meth:`reset` to start over from the beginning.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.sample_rate = int(sample_rate)

    def render(
        self,
        *,
        num_samples: int | None = None,
        duration: float | None = None,
        **params,
    ) -> np.ndarray:
        """Render the next chunk of audio as a float32 array in [-1, 1].

        The length is given by exactly one of ``num_samples`` or ``duration``
        (in seconds, rounded to the nearest sample). ``params`` are the
        algorithm-specific sound parameters.
        """
        if (num_samples is None) == (duration is None):
            raise ValueError("Exactly one of num_samples or duration must be given")
        if duration is not None:
            if duration < 0:
                raise ValueError(f"duration must be non-negative, got {duration}")
            num_samples = round(duration * self.sample_rate)
        elif num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}")
        num_samples = int(num_samples)
        samples = np.asarray(self._render(num_samples, **params), dtype=np.float32)
        if samples.shape != (num_samples,):
            raise RuntimeError(
                f"{type(self).__name__}._render returned shape {samples.shape}, "
                f"expected ({num_samples},)"
            )
        return samples

    def reset(self) -> None:
        """Reset internal state. Stateless synths need not override this."""

    @abstractmethod
    def _render(self, num_samples: int, **params) -> np.ndarray:
        """Produce exactly ``num_samples`` mono samples in [-1, 1]."""
