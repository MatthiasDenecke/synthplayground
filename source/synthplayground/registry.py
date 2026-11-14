"""Registry of synth algorithms by name.

Every module in :mod:`synthplayground.algorithms` is imported automatically on
first lookup, so dropping in a new algorithm only requires a new module there
whose class is decorated with :func:`register_synth`.
"""

import importlib
import pkgutil
import warnings

from synthplayground.synth import Synth

_REGISTRY: dict[str, type[Synth]] = {}
_discovered = False


def register_synth(name: str):
    """Class decorator registering a :class:`Synth` subclass under ``name``."""

    def decorator(cls: type[Synth]) -> type[Synth]:
        if not (isinstance(cls, type) and issubclass(cls, Synth)):
            raise TypeError(f"{cls!r} is not a subclass of Synth")
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"A synth named {name!r} is already registered: {existing!r}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True
    from synthplayground import algorithms

    for module in pkgutil.iter_modules(algorithms.__path__, algorithms.__name__ + "."):
        try:
            importlib.import_module(module.name)
        except ImportError as exc:
            # Typically an optional dependency (e.g. tensorflow) that is not installed.
            warnings.warn(f"Skipping synth module {module.name}: {exc}", stacklevel=2)


def available_synths() -> list[str]:
    """Names of all registered synth algorithms."""
    _discover()
    return sorted(_REGISTRY)


def get_synth(name: str) -> type[Synth]:
    """Look up a registered synth class by name."""
    _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown synth {name!r}; available: {available_synths()}") from None


def create_synth(name: str, **kwargs) -> Synth:
    """Instantiate a registered synth by name."""
    return get_synth(name)(**kwargs)
