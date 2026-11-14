"""Pipe organ by modal synthesis.

A pipe is described by its resonances (modes): each mode is a damped
oscillator with its own frequency and decay time, computed directly from the
acoustics of the pipe (:func:`synthplayground.organ.pipe_modes`) - the end
corrections of the open end and the mouth, which shrink with frequency and so
stretch the higher modes away from the harmonics, and the losses to the walls
and to radiation. All modes are driven
together by the air jet at the mouth (:class:`synthplayground.organ.Jet`),
which in turn is steered by the sum of their motion. This is Fletcher's
theory of the organ flue pipe as nonlinearly coupled modes (Fletcher 1974,
1976; Fletcher & Rossing, *The Physics of Musical Instruments*): the jet
locks the slightly inharmonic modes into one strictly periodic tone, and
while the pipe starts to speak the modes still ring at their own pitches.

::

             +--> mode 0 (steady flow) --+
             +--> mode 1 (f1, decay 1) --+
    drive ---+--> mode 2 (f2, decay 2) --+--> sum --> motion at the mouth --+
      ^      +--> ...                    |                                  |
      |      +--> mode M (fM, decay M) --+     |                            |
      |                                        v                            |
      |                             radiation high-pass --> radiated sound  |
      +------------------------------- jet <--------------------------------+

Compared to a waveguide the resonator costs more per sample and stops at the
highest mode included, but every mode has exactly the frequency and damping
the acoustics ask for.
"""

import cmath
import math

import numpy as np

from synthplayground.organ import (
    PRINCIPAL,
    SPEED_OF_SOUND,
    FluePipe,
    Jet,
    PipeGeometry,
    PipeOrganSynth,
    Voicing,
    Wind,
    pipe_modes,
    trailing_silence,
)
from synthplayground.registry import register_synth

MAX_MODES = 40
_HIGHEST_MODE = 0.45  # of the sample rate


class ModalPipe(FluePipe):
    """One flue pipe as a bank of modes coupled by the jet."""

    def __init__(
        self,
        design_frequency: float,
        sample_rate: int,
        noise_seed: int = 0,
        voicing: Voicing = PRINCIPAL,
    ) -> None:
        geometry = PipeGeometry.principal(design_frequency)
        self._sample_rate = sample_rate

        # Mode 0 is the steady flow through the pipe, which is open at both
        # ends: it does not oscillate and does not radiate, but the jet feels
        # it. It is what the onset of the jet sets going first.
        period = sample_rate / design_frequency
        steady_loss = geometry.wall_loss(design_frequency) + voicing.mouth_loss
        poles, weights = [cmath.exp(-steady_loss / period)], [1 / period]
        for mode in pipe_modes(geometry, voicing):
            if mode.number > MAX_MODES or mode.frequency > _HIGHEST_MODE * sample_rate:
                break
            round_trip = mode.number / mode.frequency * sample_rate  # in samples
            poles.append(cmath.exp((-mode.loss + 2j * math.pi * mode.number) / round_trip))
            weights.append(2 / round_trip)
        if len(poles) == 1:
            raise ValueError(f"no mode of a {design_frequency} Hz pipe fits the sample rate")

        self._poles = np.array(poles)
        self._modes = np.zeros(len(poles), dtype=complex)
        self._motion = self._modes.real  # a view: follows the modes
        self._weights = np.array(weights)

        # The open end reflects the low-passed wave and radiates the rest
        # (the same one-pole fit as in the waveguide, here for one end).
        s = (geometry.radius * sample_rate / SPEED_OF_SOUND) ** 2
        self._radiation = 1 - ((2 * s + 1) - math.sqrt(4 * s + 1)) / (2 * s)

        z1 = cmath.exp(-2j * math.pi / period)
        response = sum(
            w / 2 * (1 / (1 - p * z1) + 1 / (1 - p.conjugate() * z1))
            for w, p in zip(weights, poles)
        )
        self._jet = Jet.design(voicing, period, abs(response), sample_rate)
        self._jet_line = self._jet.delay_line()

        self._wind = Wind.for_pipe(design_frequency, sample_rate)
        self._rng = np.random.default_rng(noise_seed)
        self._state = (0,) + (0.0,) * 9
        self._windless = False
        self._quiet = 0
        self._rung_out = math.ceil(2 * period)  # silent for two periods

    @property
    def mode_frequencies(self) -> np.ndarray:
        """Frequencies of the oscillating modes in Hz, lowest first."""
        return np.angle(self._poles[1:]) * self._sample_rate / (2 * np.pi)

    @property
    def mode_decay_times(self) -> np.ndarray:
        """Time in seconds in which each of these modes, left alone, decays to 1/e."""
        return -1 / (np.log(np.abs(self._poles[1:])) * self._sample_rate)

    @property
    def silent(self) -> bool:
        return self._windless and self._quiet >= self._rung_out

    def strike(self, amount: float) -> None:
        self._modes += amount  # every mode takes it up, as it does the jet's drive
        self._quiet = 0

    def render(self, num_samples: int, blowing: bool) -> np.ndarray:
        jet = self._jet
        out = [0.0] * num_samples
        noise = ((self._rng.random(num_samples) - 0.5) * (2 * jet.noise)).tolist()
        modes, motion, poles, weights = self._modes, self._motion, self._poles, self._weights
        line = self._jet_line
        line_len = len(line)
        c_low, c_high, c_noise = jet.lowpass, jet.highpass, jet.noise_lowpass
        travel, slowest, scale, offset = jet.travel, jet.SLOWEST, jet.scale, jet.offset
        gain, vortex, c_rad = jet.gain, jet.vortex, self._radiation
        target = 1.0 if blowing else 0.0
        c_wind = self._wind.rise if blowing else self._wind.fall
        tanh, sqrt = math.tanh, math.sqrt
        multiply, add, dot = np.multiply, np.add, np.dot
        li, groove, wind, speed, wave, rad, deflection, turbulence, flow, drive = self._state
        delay = travel / max(speed, slowest)

        for i in range(num_samples):
            # Wind pressure in the foot and the resulting jet speed (Bernoulli).
            if wind != target:
                groove += (target - groove) * c_wind
                wind += (groove - wind) * c_wind
                if abs(wind - target) < 1e-9 and abs(groove - target) < 1e-9:
                    groove = wind = target
                speed = sqrt(wind)
                delay = travel / speed if speed > slowest else travel / slowest

            # Every mode turns by its own angle and decays.
            multiply(modes, poles, out=modes)
            wave = float(dot(weights, motion))  # motion at the mouth
            rad += c_rad * (wave - rad)  # what the open end reflects ...
            out[i] = wave - rad  # ... and what it radiates

            # The jet: displaced by the motion at the mouth, delayed by its travel.
            line[li] = wave
            position = li - delay
            if position < 0:
                position += line_len
            j0 = int(position)
            j1 = j0 + 1
            if j1 == line_len:
                j1 = 0
            delayed = line[j0] + (position - j0) * (line[j1] - line[j0])
            deflection += c_low * (delayed - deflection)
            turbulence += c_noise * (speed * noise[i] - turbulence)
            previous = flow
            flow = wind * gain * tanh(scale * deflection + turbulence + offset)
            drive = c_high * (drive + flow - previous)
            li += 1
            if li == line_len:
                li = 0

            # All modes take up the jet's drive, less the vortex loss.
            loss = vortex * abs(wave)
            add(modes, -drive - wave * loss / (1 + loss), out=modes)

        self._state = (li, groove, wind, speed, wave, rad, deflection, turbulence, flow, drive)
        result = np.array(out)
        self._quiet = trailing_silence(result, self._quiet)
        self._windless = not blowing and wind < 1e-4
        return result


@register_synth("organ_modal")
class ModalOrganSynth(PipeOrganSynth):
    """Church organ principal stops; every pipe is a bank of jet-coupled modes."""

    # Measured (synthplayground.metrics.aliasing_level, a third of a semitone apart,
    # at 22.05, 44.1 and 48 kHz): with this many samples to the period the aliasing
    # of a pipe stays below -70 dB of its tone at every pitch, with 2 to 3 dB to spare.
    samples_per_period = 70.0

    def _build_pipe(self, design_frequency: float, sample_rate: int, noise_seed: int) -> FluePipe:
        return ModalPipe(design_frequency, sample_rate, noise_seed, self.voicing)
