import numpy as np
import pytest

from synthplayground.synth import Synth


class Additive(Synth):
    """A synth whose sound is known exactly: partials, noise, linear attack and release.

    ``partials`` are ``(frequency as a multiple of the key's, amplitude)``;
    ``sharp`` mistunes everything by a factor; ``warble`` is the depth of a
    slow tremolo; with ``sustain`` the note ignores the release of the key.
    """

    def __init__(
        self,
        sample_rate=22050,
        partials=((1, 0.5),),
        noise=0.0,
        attack=0.0,
        release=0.0,
        sharp=1.0,
        warble=0.0,
        sustain=False,
        highest=None,
    ):
        super().__init__(sample_rate)
        self.partials, self.noise, self.sharp, self.warble = partials, noise, sharp, warble
        self.attack, self.release, self.sustain, self.highest = attack, release, sustain, highest
        self.reset()

    def reset(self):
        self._position = 0
        self._gain = 0.0
        self._rng = np.random.default_rng(0)

    def _render(self, num_samples, frequency=440.0, amplitude=1.0):
        if self.highest is not None and frequency > self.highest:
            raise ValueError(f"{frequency} Hz is too high")
        time = (self._position + np.arange(num_samples)) / self.sample_rate
        self._position += num_samples
        tone = sum(
            level * np.sin(2 * np.pi * multiple * frequency * self.sharp * time)
            for multiple, level in self.partials
        )
        tone = tone * (1 + self.warble * np.sin(2 * np.pi * frequency / 40 * time))
        tone = tone + self.noise * self._rng.standard_normal(num_samples)

        # The gain moves towards its target at a fixed rate: linear attack and release.
        target = 1.0 if (amplitude > 0 or self.sustain) else 0.0
        seconds = self.attack if target else self.release
        if seconds > 0:
            step = np.sign(target - self._gain) / (seconds * self.sample_rate)
            gains = np.clip(self._gain + step * np.arange(1, num_samples + 1), 0.0, 1.0)
        else:
            gains = np.full(num_samples, target)
        if num_samples:
            self._gain = float(gains[-1])
        return (amplitude if amplitude > 0 else 1.0) * gains * tone


@pytest.fixture
def additive():
    """The :class:`Additive` synth class."""
    return Additive
