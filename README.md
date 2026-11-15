# synthplayground

A playground for exploring audio synthesizer algorithms in Python.

## Setup

```bash
conda activate synthplayground          # Python 3.12
pip install -e ".[dev,dl]"              # dl = TensorFlow for deep-learning synths
conda install -c conda-forge python-sounddevice alsa-plugins   # live playback
pytest
```

### Audio playback under WSL

WSL provides sound through PulseAudio, but the conda-forge PortAudio only talks to
ALSA. `alsa-plugins` bridges the two; ALSA additionally has to be told to use it,
which is done with an environment variable stored in the conda env:

```bash
E=$CONDA_PREFIX
cp $E/etc/alsa/conf.d/99-pulseaudio-default.conf.example $E/etc/alsa/conf.d/99-pulseaudio-default.conf
conda env config vars set ALSA_CONFIG_PATH=$E/share/alsa/alsa.conf:$E/etc/alsa/conf.d/50-pulseaudio.conf:$E/etc/alsa/conf.d/99-pulseaudio-default.conf
conda activate synthplayground   # reactivate to pick it up
```

The first playback after a while can take several seconds to start.

## Listening

```python
from synthplayground import create_synth, play

melody = [
    {"duration": 0.3, "frequency": 440.0, "amplitude": 0.5},
    {"duration": 0.3, "frequency": 660.0, "amplitude": 0.5},
]
play(create_synth("sine"), melody)                      # play on the sound card
play(create_synth("sine"), melody, filename="out.wav")  # or write a 16-bit WAV
```

Each dict is passed to one `render()` call and the results are concatenated.

Synths are monophonic. To play several voices together, render each voice with its own
synth instance and mix them; `church_reverb` then puts the result into a church:

```python
from synthplayground import church_reverb, create_synth, mix, normalize, render_segments, write_wav

voices = [
    [{"duration": 2.0, "frequency": 146.83, "stops": (8, 4, 2)}],   # D3
    [{"duration": 2.0, "frequency": 220.00, "stops": (8, 4, 2)}],   # A3
]
dry = mix(render_segments(create_synth("organ_waveguide"), voice) for voice in voices)
write_wav("chord.wav", normalize(church_reverb(dry, 44100)), 44100)
```

## Pipe organ

Two synths model the flue pipes of a church organ physically, pipe by pipe. They share the
organ around the pipes (`synthplayground/organ.py`: pipe dimensions after the classical
organ-building scale, the wind in the pipe foot, the air jet at the mouth, voicing, and
the keyboard and stops) and differ in how the air column of a pipe is simulated:

| synth | approach | the air column is ... |
| --- | --- | --- |
| `organ_waveguide` | digital waveguide synthesis | a travelling wave in a delay line, closed into a loop by filters for the two open ends and the walls |
| `organ_modal` | modal synthesis (Fletcher's coupled-mode theory of the flue pipe) | a bank of resonances whose frequencies and decay times follow from the acoustics of the pipe |

In both, the tone is not put in but arises: the air jet at the mouth is pushed around by
the sound in the pipe and feeds it in turn, until losses that grow with the amplitude stop
the growth. The way a pipe starts to speak, its breath noise, the sag in pitch when the
wind is cut and the ringing out afterwards all come out of that loop. Every pipe is tuned
and regulated by "listening" to it when it is first used, and then mistuned by a cent or
so, as the ranks of a real organ are.

```python
organ = create_synth("organ_modal")            # or "organ_waveguide"
organ.render(duration=1.0, frequency=220.0)                     # 8' principal
organ.render(duration=1.0, frequency=220.0, stops=(8, 4, 2))    # principal chorus
organ.render(duration=0.5, amplitude=0.0)                       # rest: the pipes ring out
```

`frequency` is the pitch of the key, `stops` the lengths in feet of the ranks that are
drawn (a stop of `n` feet sounds `8 / n` times the key's frequency) and `amplitude` a
mixing level, where `0` means that no key is pressed. A key is held for as long as
consecutive `render()` calls ask for the same frequency and stops; pipe frequencies must
not exceed an eighth of the sample rate.

The jet of a pipe is nonlinear and makes partials without end; those above half the sample
rate would fold back as tones that do not belong (aliasing), the more so the fewer samples
a period has. Each pipe is therefore simulated at the smallest multiple of the sample rate
that gives it enough samples per period (`samples_per_period`: 90 for the waveguide, 70 for
the modal synth) and filtered down. At 44.1 kHz that is every pipe above 490 Hz or 630 Hz,
up to 11 times for the highest. It keeps the aliasing of every pipe below -70 dB of its
tone, costs computing time in proportion, and delays the oversampled pipes by 1.1 ms.

The organ synths take an optional `voicing` (how the mouths and jets of the pipes are set
up, `synthplayground.organ.PRINCIPAL` by default), e.g. to switch the breath noise off:
`create_synth("organ_modal", voicing=dataclasses.replace(PRINCIPAL, turbulence=0.0))`.

`examples/organ_comparison.py` renders the opening of Bach's Toccata in D minor with both
synths, dry and in a church, to `output/organ_{waveguide,modal}_{dry,church}.wav` (a
minute or two on a multi-core machine; the waveguide is about four times faster than the
modal synth).

## Metrics

`synthplayground.metrics` puts numbers on the output of synths without needing recordings
of the real instrument. It therefore says how two synths differ, whether a pipe model
behaves like a pipe and whether the numerics are sound, but not how real a synth sounds.

| module | question | functions |
| --- | --- | --- |
| `metrics.tone` | What does a note sound like, and how do two synths differ? | `measure_note` (pitch, level, partials, noise between the harmonics, attack, overshoot, release), `note_difference`, `band_levels` for whole renderings; building blocks such as `partial_levels`, `inharmonic_level`, `envelope` |
| `metrics.physics` | Does a pipe model behave like a pipe? | `measure_resonances` (taps a pipe and reads its resonances from the ringing), `detailed_resonances` and `ideal_resonances` (two theories to compare with), `resonance_deviations`, `power_law` (scaling over the compass); tuning errors are `Note.detuning` |
| `metrics.numerics` | Is the simulation numerically sound? | `aliasing_level`, `inharmonic_tones`, `convergence` (same note at several sample rates), `stability` (every pitch speaks, holds steady, dies away) |
| `metrics.cost` | What does it cost? | `measure_cost` (real-time factor, one-off preparation) |

```python
from synthplayground import create_synth
from synthplayground.metrics import compass, measure_note, note_difference, stability

waveguide = measure_note(create_synth("organ_waveguide"), 220.0)
modal = measure_note(create_synth("organ_modal"), 220.0, stops=(8,))
print(waveguide.partials, 1000 * waveguide.attack, waveguide.detuning)
print(note_difference(modal, waveguide))                 # modal minus waveguide

checks = stability(create_synth("organ_modal"), compass(65.4, 2093.0))
print([check for check in checks if not check.ok])       # pitches with a problem
```

The note-level functions work with any synth that takes `frequency` and `amplitude` (where
`amplitude=0` releases the note), not just the organs. `examples/organ_metrics.py` runs
everything on the two organ synths and writes the tables to `output/organ_metrics.md`
(a few minutes), among them the worst aliasing per octave against the target of -70 dB.

## Layout

```
source/synthplayground/
    synth.py            # Synth abstract base class
    registry.py         # register_synth / get_synth / create_synth / available_synths
    audio.py            # play / render_segments / mix / normalize / write_wav / read_wav
    effects.py          # church_reverb
    organ.py            # what the pipe-organ synths share
    algorithms/         # one module per synth algorithm (auto-discovered)
        sine.py
        organ_waveguide.py
        organ_modal.py
    metrics/            # assessing synth output
        tone.py         # measurements on notes, differences between synths
        physics.py      # pipe resonances against theory, scaling
        numerics.py     # aliasing, convergence, stability
        cost.py         # computing time
examples/               # scripts that render something to listen to
tests/synthplayground/  # mirrors source/synthplayground; test_<module>.py
```

## Adding a synth algorithm

1. Create `source/synthplayground/algorithms/my_synth.py`:

   ```python
   import numpy as np

   from synthplayground.registry import register_synth
   from synthplayground.synth import Synth


   @register_synth("my_synth")
   class MySynth(Synth):
       def _render(self, num_samples: int, cutoff: float = 1000.0) -> np.ndarray:
           ...  # return num_samples mono samples in [-1, 1]
   ```

   Sound parameters (here `cutoff`) are keyword arguments of `_render`; callers pass
   them to `render()`. The constructor is for setup only (sample rate,
   model weights, ...). Override `reset()` if the synth keeps state between
   `render()` calls (e.g. oscillator phase).

2. Add tests in `tests/synthplayground/algorithms/test_my_synth.py`. For TensorFlow
   based synths, start the test file with `tf = pytest.importorskip("tensorflow")`.

3. Use it:

   ```python
   from synthplayground import create_synth

   synth = create_synth("my_synth", sample_rate=44100)
   audio = synth.render(duration=2.0, cutoff=500.0)  # 2 seconds
   block = synth.render(num_samples=512, cutoff=800.0)  # continues where it left off
   ```
