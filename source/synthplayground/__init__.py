"""Playground for exploring audio synthesizer algorithms."""

from synthplayground.audio import play, write_wav
from synthplayground.registry import available_synths, create_synth, get_synth, register_synth
from synthplayground.synth import Synth

__all__ = [
    "Synth",
    "available_synths",
    "create_synth",
    "get_synth",
    "play",
    "register_synth",
    "write_wav",
]
