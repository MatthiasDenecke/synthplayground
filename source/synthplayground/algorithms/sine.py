"""Basic sine wave oscillator."""

import numpy as np

from synthplayground.registry import register_synth
from synthplayground.synth import DEFAULT_SAMPLE_RATE, Synth


@register_synth("sine")
class SineSynth(Synth):
    """Generates a sine wave; frequency and amplitude are given per render call."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        super().__init__(sample_rate)
        self._phase = 0.0  # in cycles, kept in [0, 1)

    def reset(self) -> None:
        self._phase = 0.0

    def _render(
        self, num_samples: int, frequency: float = 440.0, amplitude: float = 1.0
    ) -> np.ndarray:
        if not 0 < frequency < self.sample_rate / 2:
            raise ValueError(
                f"frequency must be in (0, {self.sample_rate / 2}) (Nyquist), got {frequency}"
            )
        if not 0 <= amplitude <= 1:
            raise ValueError(f"amplitude must be in [0, 1], got {amplitude}")
        increment = frequency / self.sample_rate
        phases = self._phase + increment * np.arange(num_samples)
        self._phase = (self._phase + increment * num_samples) % 1.0
        return amplitude * np.sin(2 * np.pi * phases)
