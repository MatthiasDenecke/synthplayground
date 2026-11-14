import sys
import types
import wave

import numpy as np
import pytest

from synthplayground.algorithms.sine import SineSynth
from synthplayground.audio import play, write_wav


@pytest.fixture
def fake_sounddevice(monkeypatch):
    calls = []
    module = types.ModuleType("sounddevice")
    module.play = lambda audio, samplerate: calls.append(("play", audio, samplerate))
    module.wait = lambda: calls.append(("wait",))
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return calls


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        frames = f.readframes(f.getnframes())
        return f.getframerate(), np.frombuffer(frames, dtype="<i2")


SEGMENTS = [
    {"duration": 0.25, "frequency": 440.0},
    {"num_samples": 1000, "frequency": 880.0, "amplitude": 0.5},
]


def test_play_writes_wav_file(tmp_path, fake_sounddevice):
    path = tmp_path / "out.wav"
    audio = play(SineSynth(sample_rate=8000), SEGMENTS, filename=path)

    reference = SineSynth(sample_rate=8000)
    expected = np.concatenate([reference.render(**s) for s in SEGMENTS])
    np.testing.assert_array_equal(audio, expected)
    assert audio.shape == (2000 + 1000,)

    rate, pcm = read_wav(path)
    assert rate == 8000
    np.testing.assert_allclose(pcm / 32767, audio, atol=1 / 32767)
    assert fake_sounddevice == []


def test_play_without_filename_uses_sounddevice(fake_sounddevice):
    audio = play(SineSynth(sample_rate=8000), [{"num_samples": 100}])
    assert len(fake_sounddevice) == 2
    (name, played, rate), wait = fake_sounddevice
    assert name == "play"
    assert played is audio
    assert rate == 8000
    assert wait == ("wait",)


def test_play_accepts_any_iterable(tmp_path):
    segments = ({"num_samples": n} for n in (10, 20))
    assert play(SineSynth(), segments, filename=tmp_path / "x.wav").shape == (30,)


def test_play_rejects_empty_segments(tmp_path):
    with pytest.raises(ValueError):
        play(SineSynth(), [], filename=tmp_path / "x.wav")


def test_play_propagates_render_errors(tmp_path):
    with pytest.raises(ValueError):
        play(SineSynth(), [{"frequency": 440.0}], filename=tmp_path / "x.wav")


def test_write_wav_clips_out_of_range_values(tmp_path):
    path = tmp_path / "clip.wav"
    write_wav(path, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 1000)
    _, pcm = read_wav(path)
    np.testing.assert_array_equal(pcm, [-32767, -32767, 0, 32767, 32767])
