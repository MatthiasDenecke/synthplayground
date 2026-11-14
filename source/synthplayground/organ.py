"""Machinery shared by the physically modelled pipe-organ synths.

A church organ is a set of *ranks* of flue pipes, one pipe per key and rank.
The synths in :mod:`synthplayground.algorithms` (``organ_waveguide`` and
``organ_modal``) differ only in how a single pipe is simulated; everything
around the pipes lives here:

* the pipe dimensions as a function of pitch (:class:`PipeGeometry`),
* the resonances the acoustics give such a pipe (:func:`pipe_modes`),
* the wind supply of a pipe (:class:`Wind`),
* the air jet at its mouth, which makes it sound (:class:`Jet`),
* the interface of a simulated pipe (:class:`FluePipe`), and running one at
  a multiple of the sample rate where that is needed (:class:`Oversampled`),
* voicing: tuning each pipe and regulating its loudness (:func:`voice`),
* the keyboard/stop logic turning ``render()`` calls into pipes that speak,
  sustain and release (:class:`PipeOrganSynth`).
"""

import cmath
import math
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from synthplayground.synth import DEFAULT_SAMPLE_RATE, Synth

SPEED_OF_SOUND = 343.0  # m/s

#: Highest pipe frequency as a fraction of the sample rate. Above this a pipe
#: has too few samples per period to be simulated faithfully.
MAX_PIPE_FREQUENCY_RATIO = 1 / 8

_REFERENCE_FREQUENCY = 440.0
_REFERENCE_RMS = 0.08  # loudness of the 440 Hz pipe at amplitude 1
_LEVEL_EXPONENT = -0.25  # trebles are voiced softer, basses fuller
_AMPLITUDE_RAMP = 0.005  # seconds over which a held note changes amplitude
_TUNING_TOLERANCE = 1e-4  # pipes are tuned to within 0.2 cents
WIND_RISE_PERIODS = 0.75  # time constant of the wind filling a pipe foot, in periods
_PLANE_WAVE_LIMIT = 3.5  # ka up to which the open-end formulas hold
_NOISE_REFERENCE_RATE = 44100  # sample rate at which Voicing.turbulence is the noise per sample


@dataclass(frozen=True)
class PipeGeometry:
    """Dimensions of an open cylindrical principal pipe sounding ``frequency``.

    The diameter follows the classical organ-building scale (Töpfer's
    *Normalmensur*): 155.5 mm at 8' C (65.4 Hz), halving on every 16th pipe,
    i.e. proportional to ``frequency ** -0.75``. Low pipes are therefore
    relatively narrow (rich in harmonics) and high pipes relatively wide.
    """

    frequency: float
    radius: float  # m
    length: float  # m, acoustic half wavelength (end corrections included)

    @classmethod
    def principal(cls, frequency: float) -> "PipeGeometry":
        radius = 0.5 * 0.1555 * (65.4 / frequency) ** 0.75
        return cls(frequency, radius, SPEED_OF_SOUND / (2 * frequency))

    def wall_loss(self, frequency: float) -> float:
        """Viscothermal amplitude loss over one round trip of the pipe."""
        alpha = 3e-5 * math.sqrt(frequency) / self.radius  # 1/m
        return 2 * self.length * alpha


@dataclass(frozen=True)
class Wind:
    """Pressure in a pipe foot (normalised to 1) after its pallet opens or closes.

    The pallet lets wind into the groove under the pipe and the groove fills
    the foot: two stages, each a first-order lag, so that the pressure sets in
    smoothly. Large pipes have large feet and fill more slowly, so the time
    constants scale with the period, within limits.
    """

    rise: float  # per-sample smoothing coefficient of each stage, pallet open
    fall: float  # ... and after it has closed

    @classmethod
    def for_pipe(cls, frequency: float, sample_rate: int) -> "Wind":
        stage = 0.5 * min(max(WIND_RISE_PERIODS / frequency, 0.0015), 0.015)
        return cls(
            rise=1 - math.exp(-1 / (stage * sample_rate)),
            fall=1 - math.exp(-1 / (1.5 * stage * sample_rate)),
        )


@dataclass(frozen=True)
class Voicing:
    """How the pipes of a rank are voiced; the same for every pipe of the rank.

    Lengths are in pipe radii, times in periods of the pipe's pitch,
    frequencies in multiples of the pitch and jet movements in jet widths, so
    that one voicing serves all sizes of pipe.
    """

    mouth_correction: float  # end correction of the mouth
    mouth_loss: float  # round-trip loss at the mouth, on top of walls and radiation
    jet_travel: float  # phase lag around the jet at the pitch
    jet_inertia: float  # the jet follows the air's displacement: low-pass corner
    jet_sharpness: float  # its flow drives the pipe by its rate of change, up to here
    jet_offset: float  # how far the lip sits off the jet's centre line
    jet_swing: float  # how far the jet swings at the lip once the tone is steady
    jet_drive: float  # gain of the jet at the onset, relative to what just starts the tone
    turbulence: float  # breath noise: random wobble of the jet (per sample at 44.1 kHz)
    turbulence_band: float  # ... band-limited to this frequency


#: The voicing of the principal stops. Both pipe models use it, so that they
#: simulate the same pipes and differ only in how they do it. The values were
#: found by fitting the tone (a strong octave, partials falling off smoothly)
#: and the speech (full tone after some 10 to 20 periods, a slight overshoot)
#: of principal pipes from 8' D to the top of the 2'.
PRINCIPAL = Voicing(
    mouth_correction=1.53,
    mouth_loss=0.031,
    jet_travel=0.487,
    jet_inertia=0.21,
    jet_sharpness=12.7,
    jet_offset=1.5,
    jet_swing=0.64,
    jet_drive=3.5,
    turbulence=0.0012,
    turbulence_band=8.0,
)


def open_end(ka: float) -> tuple[float, float]:
    """Reflection magnitude and end correction (in radii) of an unflanged open end.

    Rational approximations after Norris & Sheng (1989) to the exact solution
    of Levine & Schwinger, good up to about ka = 3.5.
    """
    magnitude = (1 + 0.2 * ka - 0.084 * ka * ka) / (1 + 0.2 * ka + 0.416 * ka * ka)
    correction = 0.6133 * (1 + 0.044 * ka * ka) / (1 + 0.19 * ka * ka)
    return magnitude, correction


@dataclass(frozen=True)
class PipeMode:
    """A resonance of the air column of a pipe."""

    number: int  # half waves along the pipe: 1 is the fundamental
    frequency: float  # Hz
    loss: float  # share of the amplitude lost per round trip through the pipe

    @property
    def decay_time(self) -> float:
        """Seconds in which the mode, left alone, decays to 1/e."""
        return self.number / (self.frequency * self.loss)


def pipe_modes(geometry: PipeGeometry, voicing: Voicing) -> Iterator[PipeMode]:
    """The modes of a pipe as its acoustics give them, lowest first.

    Mode ``n`` fits ``n`` half waves into the length of the pipe plus the end
    corrections of the open end and the mouth. Both corrections shrink with
    frequency (the mouth is a mass of air: its correction fades as
    ``atan(k m) / k``), which stretches the higher modes away from the
    harmonics of the first, which sits on ``geometry.frequency``. Each mode
    loses amplitude to the walls, to radiation from the open end and from the
    mouth, and to the mouth itself (``voicing.mouth_loss``). The series ends
    where the pipe is too wide for plane waves and the formulas stop holding.
    """
    radius = geometry.radius
    mouth = voicing.mouth_correction * radius

    def correction(k: float) -> float:
        return open_end(k * radius)[1] * radius + math.atan(k * mouth) / k

    # Body length that puts the first mode on the design frequency.
    k1 = 2 * math.pi * geometry.frequency / SPEED_OF_SOUND
    body = math.pi / k1 - correction(k1)
    number = 0
    while True:
        number += 1
        k = number * k1
        for _ in range(30):  # fixed-point iteration, converges quickly
            k = number * math.pi / (body + correction(k))
        frequency = k * SPEED_OF_SOUND / (2 * math.pi)
        ka = k * radius
        if ka > _PLANE_WAVE_LIMIT:
            return
        loss = (
            geometry.wall_loss(frequency)
            - math.log(open_end(ka)[0])
            + 0.5 * ka * ka / (1 + (k * mouth) ** 2)  # radiation from the mouth
            + voicing.mouth_loss
        )
        yield PipeMode(number, frequency, loss)


@dataclass(frozen=True)
class Jet:
    """The air jet of one pipe, as coefficients for the simulation loops.

    Both pipe models share this physics (after Fletcher, Coltman and Verge):
    the sound at the mouth displaces the jet (``lowpass``), the displacement
    travels up the jet (``travel`` samples at full wind, longer while the wind
    builds up), and the flow that the lip lets into the pipe,
    ``gain * tanh(scale * displacement + offset)``, drives the pipe by its
    rate of change (``highpass``). The tone stops growing partly because the
    jet saturates but mostly because the sound sheds vortices at the lip, a
    loss of ``vortex * |v|`` per unit of sound that grows with the amplitude
    (applied as ``v / (1 + vortex * |v|)``); it is set so that
    the tone settles when the jet swings by ``Voicing.jet_swing``, which keeps
    the flow smooth and the tone free of harshness.
    """

    lowpass: float
    highpass: float
    travel: float
    scale: float
    offset: float
    gain: float
    vortex: float
    noise: float
    noise_lowpass: float

    #: Jet speed (relative to full wind) below which the travel time is capped.
    SLOWEST = 0.25

    @classmethod
    def design(
        cls,
        voicing: Voicing,
        period: float,
        resonance: float,
        sample_rate: int = _NOISE_REFERENCE_RATE,
    ) -> "Jet":
        """Jet for a pipe with ``period`` samples per cycle.

        ``resonance`` is how much the pipe amplifies a drive at its pitch (the
        mouth wave per unit of drive); the jet gain that just starts the tone
        is its inverse. ``sample_rate`` matters for the breath noise only: a
        random number per sample is noise up to half the sample rate, so its
        size grows with the square root of the rate to keep the noise within
        the band of the pipe the same at every rate.
        """
        omega = 2 * math.pi / period
        z1 = cmath.exp(-1j * omega)
        c_low = 1 - math.exp(-omega * voicing.jet_inertia)
        c_high = math.exp(-omega * voicing.jet_sharpness)
        lowpass = c_low / (1 - (1 - c_low) * z1)
        highpass = c_high * (1 - z1) / (1 - c_high * z1)
        # Phase lag of the filters counts towards the travel time.
        travel = voicing.jet_travel * period + cmath.phase(lowpass * highpass) / omega
        onset = voicing.jet_drive / resonance
        steady = onset * _describing_gain(voicing.jet_swing, voicing.jet_offset)
        return cls(
            lowpass=c_low,
            highpass=c_high,
            travel=max(0.0, travel),
            scale=1 / abs(lowpass),
            offset=voicing.jet_offset,
            gain=onset * math.cosh(voicing.jet_offset) ** 2 / abs(highpass),
            vortex=_vortex_strength(steady - 1 / resonance) / voicing.jet_swing,
            noise=voicing.turbulence * math.sqrt(sample_rate / _NOISE_REFERENCE_RATE),
            noise_lowpass=1 - math.exp(-min(omega * voicing.turbulence_band, 3.0)),
        )

    def delay_line(self) -> list[float]:
        """Buffer long enough for the slowest jet."""
        return [0.0] * (math.ceil(self.travel / self.SLOWEST) + 3)


def _describing_gain(swing: float, offset: float) -> float:
    """Gain of the jet for a tone swinging it by ``swing``, relative to a faint tone."""
    phases = [2 * math.pi * (i + 0.5) / 64 for i in range(64)]
    fundamental = sum(math.tanh(swing * math.sin(p) + offset) * math.sin(p) for p in phases) / 32
    return fundamental / swing * math.cosh(offset) ** 2


def _vortex_strength(loss: float) -> float:
    """``vortex * amplitude`` at which a tone loses the fraction ``loss`` per round trip.

    Inverts the fundamental component of ``v * x / (1 + x)`` with
    ``x = strength * |sin|`` by bisection.
    """
    phases = [math.pi * (i + 0.5) / 64 for i in range(64)]

    def lost(strength: float) -> float:
        return sum(
            math.sin(p) ** 2 * strength * math.sin(p) / (1 + strength * math.sin(p))
            for p in phases
        ) / 32

    if loss <= 0:
        return 0.0
    low, high = 0.0, 1.0
    while lost(high) < loss and high < 1e6:
        high *= 2
    for _ in range(50):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if lost(middle) < loss else (low, middle)
    return high


class FluePipe(ABC):
    """A single simulated organ pipe; stateful like a :class:`Synth`."""

    @abstractmethod
    def render(self, num_samples: int, blowing: bool) -> np.ndarray:
        """Next ``num_samples`` of radiated sound; ``blowing`` = pallet open."""

    @property
    @abstractmethod
    def silent(self) -> bool:
        """True once the wind is off and the pipe has rung out."""

    @abstractmethod
    def strike(self, amount: float) -> None:
        """Tap the pipe: put a pulse of size ``amount`` into its air column.

        With the wind off, what is rendered next is the pipe ringing at its
        own resonances, which is how they can be measured. Keep ``amount``
        small (the steady tone is of order 1) to stay clear of the losses
        that grow with the amplitude.
        """


_SILENCE = 1e-6  # a pipe is taken to be silent below this


def trailing_silence(sound: np.ndarray, before: int, threshold: float = _SILENCE) -> int:
    """Number of silent samples at the end of a pipe's output so far.

    ``before`` is the count up to the previous block. Pipes use this to tell
    when they have rung out: silent for a couple of periods, not merely
    passing through zero.
    """
    loud = np.flatnonzero(np.abs(sound) > threshold)
    if len(loud) == 0:
        return before + len(sound)
    return len(sound) - 1 - int(loud[-1])


_DECIMATION_DELAY = 48  # samples at the output rate; 1.1 ms at 44.1 kHz


@lru_cache(maxsize=None)
def _decimation_filter(factor: int) -> np.ndarray:
    """Low-pass for bringing a signal down to 1 / ``factor`` of its sample rate.

    A Kaiser-windowed sinc, flat up to 0.45 of the new rate (19.8 kHz at
    44.1 kHz) and more than 70 dB down from half the new rate on, so that
    nothing above it is left to fold back. It is symmetric and
    ``2 * _DECIMATION_DELAY * factor + 1`` long, i.e. it delays by
    ``_DECIMATION_DELAY`` samples at the new rate whatever the factor.
    """
    position = np.arange(-_DECIMATION_DELAY * factor, _DECIMATION_DELAY * factor + 1)
    taps = np.sinc(0.95 * position / factor) * np.kaiser(len(position), 7.3)
    return taps / taps.sum()


class Oversampled(FluePipe):
    """A pipe simulated at ``factor`` times the sample rate and filtered down to it.

    The jet of a pipe is nonlinear: it makes partials without end, and those
    above half the sample rate come back as tones that do not belong
    (aliasing). The fewer samples there are to a period, the stronger they
    are, so high pipes are simulated at a multiple of the sample rate, where
    the partials have room to die away before they reach the limit, and only
    what is audible is handed on. The price is ``factor`` times the computing
    and a delay of ``_DECIMATION_DELAY`` samples.
    """

    def __init__(self, pipe: FluePipe, factor: int) -> None:
        if factor < 1:
            raise ValueError(f"factor must be at least 1, got {factor}")
        self._pipe, self._factor = pipe, factor
        self._taps = _decimation_filter(factor)
        self._history = np.zeros(len(self._taps) - 1)  # samples the filter still covers

    def render(self, num_samples: int, blowing: bool) -> np.ndarray:
        if num_samples == 0:
            return np.zeros(0)
        fine = self._pipe.render(num_samples * self._factor, blowing)
        signal = np.concatenate([self._history, fine])
        self._history = signal[len(fine) :]
        windows = np.lib.stride_tricks.sliding_window_view(signal, len(self._taps))
        return windows[:: self._factor] @ self._taps

    @property
    def silent(self) -> bool:
        return self._pipe.silent and not np.any(np.abs(self._history) > _SILENCE)

    def strike(self, amount: float) -> None:
        self._pipe.strike(amount)


class PipeDoesNotSpeakError(RuntimeError):
    """The pipe model does not settle on its fundamental at this pitch."""


def measure_tone(signal: np.ndarray, sample_rate: int, nominal: float) -> tuple[float, float]:
    """Frequency (Hz) and amplitude of the partial of ``signal`` near ``nominal``.

    The signal is correlated with a windowed complex exponential at
    ``nominal`` in two frames a few periods apart; the phase advance between
    the frames gives the frequency to a small fraction of a cent.
    """
    hop = max(1, round(4 * sample_rate / nominal))
    n = len(signal) - hop
    if n < 8 * sample_rate / nominal:
        raise ValueError("signal too short to measure its frequency")
    window = np.hanning(n)
    reference = window * np.exp(-2j * np.pi * nominal / sample_rate * np.arange(n))
    first = np.dot(signal[:n], reference)
    second = np.dot(signal[hop : hop + n], reference)
    advance = np.angle(second * np.conj(first))  # = 2 pi f hop / sample_rate (mod 2 pi)
    expected = 2 * np.pi * nominal * hop / sample_rate
    deviation = (advance - expected + np.pi) % (2 * np.pi) - np.pi
    frequency = nominal + deviation * sample_rate / (2 * np.pi * hop)
    amplitude = 2 * abs(first) / window.sum()
    return float(frequency), float(amplitude)


def voice(
    build: Callable[[float], FluePipe], frequency: float, sample_rate: int
) -> tuple[float, float]:
    """Tune and regulate a pipe the way an organ builder does: by listening.

    ``build(design_frequency)`` must return a fresh pipe. The sounding pitch
    of a flue pipe differs from its design (resonance) frequency because the
    air jet pulls it, so the pipe is blown until it is steady, its pitch and
    loudness are measured, and the result is the pair ``(design_frequency,
    gain)`` that makes it sound exactly ``frequency`` at its regulated level.
    """
    period = sample_rate / frequency
    settle = round(max(0.15 * sample_rate, 70 * period))
    measure = round(max(0.03 * sample_rate, 30 * period))
    design = frequency
    for _ in range(3):  # one correction is enough for all but the highest pipes
        tone = build(design).render(settle + measure, blowing=True)[settle:]
        sounding, fundamental = measure_tone(tone, sample_rate, frequency)
        rms = float(np.sqrt(np.mean(tone**2)))
        if not rms > 0 or fundamental < 0.2 * rms or abs(sounding / frequency - 1) > 0.08:
            raise PipeDoesNotSpeakError(
                f"pipe designed for {design:.1f} Hz does not speak "
                f"the fundamental {frequency:.1f} Hz"
            )
        if abs(sounding / frequency - 1) < _TUNING_TOLERANCE:
            break
        design *= frequency / sounding
    level = _REFERENCE_RMS * (frequency / _REFERENCE_FREQUENCY) ** _LEVEL_EXPONENT
    return design, level / rms


class _Note:
    """The pipes sounding for one key: one per drawn stop."""

    def __init__(self, key: tuple, pipes: list[tuple[FluePipe, float]], amplitude: float) -> None:
        self.key = key
        self.pipes = pipes  # (pipe, gain)
        self.amplitude = amplitude
        self.held = True

    def render(self, num_samples: int, amplitude: float, ramp: int) -> np.ndarray:
        out = np.zeros(num_samples)
        for pipe, gain in self.pipes:
            out += gain * pipe.render(num_samples, self.held)
        if self.held and amplitude != self.amplitude:
            envelope = np.full(num_samples, amplitude)
            ramp = min(ramp, num_samples)
            envelope[:ramp] = np.linspace(self.amplitude, amplitude, ramp, endpoint=False)
            self.amplitude = amplitude
            return out * envelope
        return out * self.amplitude

    @property
    def silent(self) -> bool:
        return not self.held and all(pipe.silent for pipe, _ in self.pipes)


class PipeOrganSynth(Synth):
    """Base class of the pipe-organ synths: one keyboard voice with stops.

    Sound parameters of ``render()``:

    ``frequency``
        Pitch of the key in Hz, as it sounds on an 8' stop.
    ``amplitude``
        Level in [0, 1]. ``0`` means no key is pressed (a rest): the pipes
        lose their wind and ring out. Organ pipes have no dynamics, so this
        is a mixing level, not a playing force.
    ``stops``
        The drawn stops as pipe lengths in feet. ``(8,)`` is a single
        principal at written pitch, ``(8, 4, 2)`` the principal chorus with
        ranks one and two octaves higher. A stop of ``n`` feet sounds
        ``8 / n`` times the key's frequency.

    As on a real organ, consecutive ``render()`` calls with the same
    frequency and stops hold the key down; a different frequency releases it
    and presses the new one, while the old pipes ring out underneath. To
    repeat a note, put a short rest between the two.

    Pipes that are high for the sample rate are simulated at a multiple of it
    (see :class:`Oversampled` and ``samples_per_period``), which costs time in
    proportion and delays them by 1.1 ms at 44.1 kHz.

    The pipes are voiced as ``voicing`` says (the principal stops by default).
    Every pipe is tuned and regulated by :func:`voice` when it first speaks
    (cached per class, sample rate and voicing), then detuned by a random amount
    within ``tuning_spread_cents``, as real ranks are never perfectly in tune
    with each other. The mistuning of a pipe is fixed by ``seed``, its stop
    and its pitch: synths with the same seed are the same organ, so several
    of them can play the voices of one piece.
    """

    #: Samples that every pipe gets per period of its pitch, by oversampling
    #: where the sample rate gives it fewer. Subclasses set it to what their
    #: pipe model needs to keep its aliasing inaudible; 0 means never to
    #: oversample.
    samples_per_period: float = 0.0

    _tunings: dict[tuple, tuple[float, float]] = {}

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        tuning_spread_cents: float = 1.5,
        seed: int = 0,
        voicing: Voicing = PRINCIPAL,
    ) -> None:
        super().__init__(sample_rate)
        if tuning_spread_cents < 0:
            raise ValueError(f"tuning_spread_cents must be >= 0, got {tuning_spread_cents}")
        self.tuning_spread_cents = float(tuning_spread_cents)
        self.seed = int(seed)
        self.voicing = voicing
        self.reset()

    @classmethod
    def forget_tunings(cls) -> None:
        """Forget how the pipes were tuned and regulated; it is redone when they next speak."""
        PipeOrganSynth._tunings.clear()

    def reset(self) -> None:
        self._notes: list[_Note] = []
        self._pipes_built = 0

    def oversampling(self, frequency: float) -> int:
        """Multiple of the sample rate at which the pipe sounding ``frequency`` is simulated."""
        return max(1, math.ceil(self.samples_per_period * frequency / self.sample_rate))

    @abstractmethod
    def _build_pipe(self, design_frequency: float, sample_rate: int, noise_seed: int) -> FluePipe:
        """A fresh, unregulated pipe whose resonances are scaled to ``design_frequency``.

        It is simulated at ``sample_rate``, which may be a multiple of the synth's.
        """

    def _pipe(self, frequency: float, design_frequency: float, noise_seed: int) -> FluePipe:
        """The pipe that is to sound ``frequency``, at the synth's sample rate."""
        factor = self.oversampling(frequency)
        pipe = self._build_pipe(design_frequency, factor * self.sample_rate, noise_seed)
        return Oversampled(pipe, factor) if factor > 1 else pipe

    def _tuning(self, frequency: float) -> tuple[float, float]:
        key = (type(self), self.sample_rate, self.voicing, frequency)
        if key not in self._tunings:
            self._tunings[key] = voice(
                lambda design: self._pipe(frequency, design, 0), frequency, self.sample_rate
            )
        return self._tunings[key]

    def _speak(self, frequency: float, stops: tuple[float, ...], amplitude: float) -> _Note:
        pipes = []
        for stop in stops:
            pitch = frequency * 8 / stop
            # Each pipe of each rank has its own fixed mistuning.
            identity = zlib.crc32(f"{self.seed}/{stop!r}/{pitch!r}".encode())
            cents = np.random.default_rng(identity).uniform(-1, 1) * self.tuning_spread_cents
            design, gain = self._tuning(pitch)
            self._pipes_built += 1
            pipe = self._pipe(
                pitch, design * 2 ** (cents / 1200), identity + 7919 * self._pipes_built
            )
            pipes.append((pipe, gain))
        return _Note((frequency, stops), pipes, amplitude)

    def _render(
        self,
        num_samples: int,
        frequency: float = 440.0,
        amplitude: float = 1.0,
        stops: Iterable[float] = (8,),
    ) -> np.ndarray:
        if not 0 <= amplitude <= 1:
            raise ValueError(f"amplitude must be in [0, 1], got {amplitude}")
        stops = tuple(float(stop) for stop in stops)
        if not stops or min(stops) <= 0:
            raise ValueError(f"stops must be a non-empty sequence of positive lengths, got {stops}")
        highest = frequency * 8 / min(stops)
        limit = self.sample_rate * MAX_PIPE_FREQUENCY_RATIO
        if not (frequency > 0 and highest <= limit):
            raise ValueError(
                f"pipe frequencies must be in (0, {limit}] Hz at this sample rate, "
                f"got up to {highest} Hz for frequency={frequency} and stops={stops}"
            )

        key = (frequency, stops) if amplitude > 0 else None
        pressed = None
        for note in self._notes:
            note.held = note.key == key
            if note.held:
                pressed = note
        if key is not None and pressed is None:
            self._notes.append(self._speak(frequency, stops, amplitude))

        out = np.zeros(num_samples)
        ramp = max(1, round(_AMPLITUDE_RAMP * self.sample_rate))
        for note in self._notes:
            out += note.render(num_samples, amplitude, ramp)
        self._notes = [note for note in self._notes if not note.silent]
        return np.clip(out, -1.0, 1.0)
