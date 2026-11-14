"""Playground for exploring audio synthesizer algorithms."""

from synthplayground.audio import mix, normalize, play, read_wav, render_segments, write_wav
from synthplayground.effects import church_reverb
from synthplayground.registry import available_synths, create_synth, get_synth, register_synth
from synthplayground.synth import Synth

__all__ = [
    "Synth",
    "available_synths",
    "church_reverb",
    "create_synth",
    "get_synth",
    "mix",
    "normalize",
    "play",
    "read_wav",
    "register_synth",
    "render_segments",
    "write_wav",
]
