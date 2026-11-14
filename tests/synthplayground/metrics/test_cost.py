import time

import numpy as np
import pytest

from synthplayground.metrics.cost import Cost, measure_cost
from synthplayground.synth import Synth

SEGMENTS = [{"duration": 0.5}, {"num_samples": 4000}]


class Slow(Synth):
    """Takes a fixed time per render call, and longer for the very first one of all."""

    prepared = False
    built = 0

    def __init__(self, per_call=0.01, preparation=0.0):
        super().__init__(8000)
        self.per_call, self.preparation = per_call, preparation
        Slow.built += 1

    def _render(self, num_samples):
        if not Slow.prepared:
            time.sleep(self.preparation)
            Slow.prepared = True
        time.sleep(self.per_call)
        return np.zeros(num_samples)


@pytest.fixture(autouse=True)
def fresh():
    Slow.prepared = False
    Slow.built = 0


def test_cost_properties():
    cost = Cost(audio=3.0, first=2.0, render=0.5)
    assert cost.realtime_factor == pytest.approx(6.0)
    assert cost.preparation == pytest.approx(1.5)
    assert Cost(audio=3.0, first=0.4, render=0.5).preparation == 0.0  # never negative


def test_measure_cost_times_the_rendering():
    cost = measure_cost(lambda: Slow(per_call=0.02), SEGMENTS)
    assert cost.audio == pytest.approx(0.5 + 4000 / 8000)
    assert 0.04 <= cost.render < 0.2  # two calls of 20 ms
    assert cost.realtime_factor == pytest.approx(cost.audio / cost.render)
    assert cost.preparation < 0.1


def test_a_slower_synth_has_a_lower_realtime_factor():
    fast = measure_cost(lambda: Slow(per_call=0.005), SEGMENTS)
    slow = measure_cost(lambda: Slow(per_call=0.05), SEGMENTS)
    assert slow.realtime_factor < fast.realtime_factor / 3


def test_what_is_done_once_counts_as_preparation():
    cost = measure_cost(lambda: Slow(per_call=0.005, preparation=0.1), SEGMENTS)
    assert cost.first >= 0.1
    assert 0.08 < cost.preparation < 0.25
    assert cost.render < 0.08


def test_every_rendering_gets_a_fresh_synth():
    measure_cost(Slow, SEGMENTS, repeats=4)
    assert Slow.built == 5  # the first and four repeats
    measure_cost(Slow, iter(SEGMENTS), repeats=1)  # any iterable, used for every rendering
    assert Slow.built == 7


def test_invalid_arguments():
    with pytest.raises(ValueError):
        measure_cost(Slow, SEGMENTS, repeats=0)
    with pytest.raises(ValueError):
        measure_cost(Slow, [])
