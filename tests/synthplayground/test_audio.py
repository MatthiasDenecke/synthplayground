import sys
import types
import wave

import numpy as np
import pytest

from synthplayground.algorithms.sine import SineSynth
from synthplayground.audio import mix, normalize, play, render_segments, write_wav
from synthplayground.audio import read_wav as read_audio


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


def test_render_segments_concatenates(tmp_path):
    audio = render_segments(SineSynth(sample_rate=8000), SEGMENTS)
    reference = SineSynth(sample_rate=8000)
    np.testing.assert_array_equal(audio, np.concatenate([reference.render(**s) for s in SEGMENTS]))
    with pytest.raises(ValueError):
        render_segments(SineSynth(), [])


def test_mix_sums_tracks_and_pads_the_shorter_ones():
    mixed = mix([np.array([0.1, 0.2, 0.3]), np.array([0.5]), np.zeros(0)])
    assert mixed.dtype == np.float32
    np.testing.assert_allclose(mixed, [0.6, 0.2, 0.3])


def test_mix_plays_voices_together():
    voices = [[{"num_samples": 800, "frequency": f, "amplitude": 0.4}] for f in (500.0, 750.0)]
    mixed = mix(render_segments(SineSynth(sample_rate=8000), voice) for voice in voices)
    spectrum = np.abs(np.fft.rfft(mixed))
    assert sorted(np.argsort(spectrum)[-2:] * 10) == [500, 750]


def test_mix_rejects_bad_input():
    with pytest.raises(ValueError):
        mix([])
    with pytest.raises(ValueError):
        mix([np.zeros((2, 2))])


def test_normalize():
    out = normalize(np.array([0.1, -0.4, 0.2]), peak=0.8)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.2, -0.8, 0.4], rtol=1e-6)
    np.testing.assert_allclose(np.abs(normalize(np.array([3.0, -5.0]))).max(), 0.9)
    assert not normalize(np.zeros(4)).any()
    assert normalize(np.zeros(0)).shape == (0,)
    with pytest.raises(ValueError):
        normalize(np.ones(3), peak=0.0)
    with pytest.raises(ValueError):
        normalize(np.ones(3), peak=1.5)


def test_read_wav_returns_what_write_wav_wrote(tmp_path):
    audio = SineSynth(sample_rate=8000).render(num_samples=500, frequency=300.0, amplitude=0.7)
    write_wav(tmp_path / "tone.wav", audio, 8000)
    read, sample_rate = read_audio(tmp_path / "tone.wav")
    assert sample_rate == 8000
    assert read.dtype == np.float32 and read.shape == (500,)
    np.testing.assert_allclose(read, audio, atol=1 / 32767)
    assert np.abs(read).max() <= 1.0


@pytest.mark.parametrize("channels, width", [(2, 2), (1, 1), (1, 3)])
def test_read_wav_rejects_other_formats(tmp_path, channels, width):
    path = tmp_path / "other.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(8000)
        f.writeframes(bytes(channels * width * 10))
    with pytest.raises(ValueError, match="16-bit mono"):
        read_audio(path)
