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

## Layout

```
source/synthplayground/
    synth.py            # Synth abstract base class
    registry.py         # register_synth / get_synth / create_synth / available_synths
    algorithms/         # one module per synth algorithm (auto-discovered)
        sine.py
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
