"""Pipe organ by digital waveguide synthesis.

The air column of a pipe is simulated as a travelling wave: a delay line as
long as one round trip through the pipe, closed into a loop by filters that
stand for what happens to the wave at the two open ends and along the walls.
The loop is driven by the air jet that blows across the mouth
(:class:`synthplayground.organ.Jet`). This is the jet-drive waveguide model
of flutes and flue pipes (McIntyre, Schumacher & Woodhouse 1983; Karjalainen
& Välimäki; Cook; Verge).

::

    +--> bore delay --> radiation --> mouth --> tuning --> x loss --+--> (+) --+
    |                  low-pass   all-pass  all-pass               |     ^     |
    |                      |                                       v     | -   |
    |                      v                                      jet ---+     |
    |               radiated sound                                             |
    +--------------------------------------------------------------------------+

Because the resonator is a delay line, the pipe has all its resonances up to
half the sample rate at the cost of a handful of operations per sample; the
price is that the physics of the ends has to be squeezed into low-order
filters.
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
    trailing_silence,
)
from synthplayground.registry import register_synth



def _phase_delay(response: complex, omega: float) -> float:
    """Phase delay in samples of a filter with ``response`` at ``omega`` rad/sample."""
    return -cmath.phase(response) / omega


class WaveguidePipe(FluePipe):
    """One flue pipe as a jet-driven digital waveguide."""

    def __init__(
        self,
        design_frequency: float,
        sample_rate: int,
        noise_seed: int = 0,
        voicing: Voicing = PRINCIPAL,
    ) -> None:
        geometry = PipeGeometry.principal(design_frequency)
        period = sample_rate / design_frequency
        omega = 2 * math.pi / period
        z1 = cmath.exp(-1j * omega)

        # Open ends: the reflected wave is a low-passed (and inverted) copy of
        # the incoming one; what is not reflected is radiated. One one-pole
        # low-pass stands for both ends: |H| ~ 1 - (ka)^2 at low frequencies.
        s = 2 * (geometry.radius * sample_rate / SPEED_OF_SOUND) ** 2
        self._radiation = ((2 * s + 1) - math.sqrt(4 * s + 1)) / (2 * s)  # pole
        radiation = (1 - self._radiation) / (1 - self._radiation * z1)

        # Mouth: a small opening, i.e. a mass of air. Reflection from a mass
        # is an all-pass; it makes the pipe look shorter to high partials, so
        # the resonances are stretched relative to the harmonics.
        mass = voicing.mouth_correction * geometry.radius * sample_rate / SPEED_OF_SOUND
        self._mouth = (1 - 2 * mass) / (1 + 2 * mass)
        mouth = (self._mouth + z1) / (1 + self._mouth * z1)

        self._loss = math.exp(-geometry.wall_loss(design_frequency)) * (1 - voicing.mouth_loss)

        # The rest of the round trip is a delay line plus a fractional-delay
        # all-pass, sized so that the loop resonates at the design frequency.
        remaining = period - _phase_delay(radiation, omega) - _phase_delay(mouth, omega)
        self._bore = [0.0] * max(1, math.floor(remaining - 0.4))
        fraction = remaining - len(self._bore)
        self._tuning = (1 - fraction) / (1 + fraction)

        # At resonance a drive comes back, round trip after round trip, as
        # loop / (1 - loop) times itself.
        loop = abs(radiation) * self._loss
        self._jet = Jet.design(voicing, period, loop / (1 - loop), sample_rate)
        self._jet_line = self._jet.delay_line()

        self._wind = Wind.for_pipe(design_frequency, sample_rate)
        self._rng = np.random.default_rng(noise_seed)
        self._state = (0, 0) + (0.0,) * 10
        self._windless = False
        self._quiet = 0
        self._rung_out = math.ceil(2 * period)  # silent for two periods

    @property
    def silent(self) -> bool:
        return self._windless and self._quiet >= self._rung_out

    def strike(self, amount: float) -> None:
        self._bore[self._state[0]] += amount  # joins the wave about to reach the mouth
        self._quiet = 0

    def render(self, num_samples: int, blowing: bool) -> np.ndarray:
        jet = self._jet
        out = [0.0] * num_samples
        noise = ((self._rng.random(num_samples) - 0.5) * (2 * jet.noise)).tolist()
        bore, line = self._bore, self._jet_line
        bore_len, line_len = len(bore), len(line)
        c_rad, c_mouth, c_tune, loss = 1 - self._radiation, self._mouth, self._tuning, self._loss
        c_low, c_high, c_noise = jet.lowpass, jet.highpass, jet.noise_lowpass
        travel, slowest, scale, offset = jet.travel, jet.SLOWEST, jet.scale, jet.offset
        gain, vortex = jet.gain, jet.vortex
        target = 1.0 if blowing else 0.0
        c_wind = self._wind.rise if blowing else self._wind.fall
        tanh, sqrt = math.tanh, math.sqrt
        bi, li, groove, wind, speed, rad, s_mouth, s_tune = self._state[:8]
        deflection, turbulence, flow, drive = self._state[8:]
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

            # The wave that left one round trip ago comes back to the mouth.
            wave = bore[bi]
            rad += c_rad * (wave - rad)  # what the open ends reflect ...
            out[i] = wave - rad  # ... and what they radiate
            mouth = c_mouth * rad + s_mouth
            s_mouth = rad - c_mouth * mouth
            tuned = c_tune * mouth + s_tune
            s_tune = mouth - c_tune * tuned
            wave = loss * tuned

            # The jet: displaced by the wave, delayed by its travel to the lip.
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

            # Back into the bore, less the vortex loss and the jet's drive.
            bore[bi] = wave / (1 + vortex * abs(wave)) - drive
            bi += 1
            if bi == bore_len:
                bi = 0
            li += 1
            if li == line_len:
                li = 0

        self._state = (bi, li, groove, wind, speed, rad, s_mouth, s_tune) + (
            deflection,
            turbulence,
            flow,
            drive,
        )
        result = np.array(out)
        self._quiet = trailing_silence(result, self._quiet)
        self._windless = not blowing and wind < 1e-4
        return result


@register_synth("organ_waveguide")
class WaveguideOrganSynth(PipeOrganSynth):
    """Church organ principal stops; every pipe is a jet-driven digital waveguide."""

    # Measured (synthplayground.metrics.aliasing_level, a third of a semitone apart,
    # at 22.05, 44.1 and 48 kHz): with this many samples to the period the aliasing
    # of a pipe stays below -70 dB of its tone at every pitch, with 2 to 3 dB to spare.
    samples_per_period = 90.0

    def _build_pipe(self, design_frequency: float, sample_rate: int, noise_seed: int) -> FluePipe:
        return WaveguidePipe(design_frequency, sample_rate, noise_seed, self.voicing)
