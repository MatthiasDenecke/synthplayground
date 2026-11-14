"""Synth algorithm implementations.

Each module here is auto-discovered by :mod:`synthplayground.registry`. To add
an algorithm, create a module with a :class:`~synthplayground.synth.Synth`
subclass decorated with ``@register_synth("<name>")``. Algorithms needing
TensorFlow should import it at module level; if TensorFlow is not installed the
module is skipped with a warning.
"""
