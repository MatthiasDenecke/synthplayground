"""Assess the two physically modelled pipe organs with numbers.

Runs the metrics of :mod:`synthplayground.metrics` on ``organ_waveguide`` and
``organ_modal``, prints the results as tables and writes them as Markdown::

    python examples/organ_metrics.py            # -> output/organ_metrics.md
    python examples/organ_metrics.py some/dir

It takes a few minutes. The octave-band comparison of the whole piece
needs the WAV files of ``examples/organ_comparison.py`` in the same directory
and is left out if they are not there.

The metrics need no recordings of a real organ, so they say how the two
synths differ, whether they behave like pipes, and whether the numerics are
sound - not how real either of them sounds.
"""

import dataclasses
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from synthplayground import create_synth, read_wav
from synthplayground.algorithms.organ_modal import ModalPipe
from synthplayground.algorithms.organ_waveguide import WaveguidePipe
from synthplayground.metrics import (
    OCTAVE_BANDS,
    aliasing_level,
    band_levels,
    compass,
    convergence,
    detailed_resonances,
    ideal_resonances,
    inharmonic_tones,
    level_db,
    measure_cost,
    measure_note,
    measure_notes,
    measure_resonances,
    note_difference,
    power_law,
    resonance_deviations,
    stability,
)
from synthplayground.organ import MAX_PIPE_FREQUENCY_RATIO, PRINCIPAL, PipeOrganSynth

SAMPLE_RATE = 44100
SYNTHS = {"organ_waveguide": WaveguidePipe, "organ_modal": ModalPipe}  # synth: its pipe
NOTES = {"A2": 110.0, "A3": 220.0, "A4": 440.0, "A5": 880.0, "A6": 1760.0, "A7": 3520.0}
RESONANCE_NOTES = {"A2": 110.0, "A4": 440.0, "A6": 1760.0}
RESONANCES = 6  # per pipe
# Every pipe from the bottom of a 16' rank (C1) up to the highest the sample rate allows.
COMPASS = compass(32.703, SAMPLE_RATE * MAX_PIPE_FREQUENCY_RATIO)
SAMPLE_RATES = (44100, 88200, 176400)
SILENT_JET = dataclasses.replace(PRINCIPAL, turbulence=0.0)  # no breath noise
ALIASING_TARGET = -70  # dB: what the synths oversample their higher pipes for


def organ(name: str, sample_rate: int = SAMPLE_RATE, **kwargs):
    """The synth without the random mistuning of its pipes, to see the model itself."""
    return create_synth(name, sample_rate=sample_rate, tuning_spread_cents=0.0, **kwargs)


def table(headers: list[str], rows: list[list]) -> list[str]:
    """A Markdown table; numbers are right-aligned."""
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(h), *(len(row[i]) for row in cells)) for i, h in enumerate(headers)]
    numeric = [all(_is_number(row[i]) for row in cells) for i in range(len(headers))]

    def line(row: list[str]) -> str:
        padded = [c.rjust(w) if n else c.ljust(w) for c, w, n in zip(row, widths, numeric)]
        return "| " + " | ".join(padded) + " |"

    rule = ["-" * (w - 1) + ":" if n else "-" * w for w, n in zip(widths, numeric)]
    return [line(headers), "| " + " | ".join(rule) + " |", *(line(row) for row in cells), ""]


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return text == "-"  # a number that is not there
    return True


def model_against_model(directory: Path) -> list[str]:
    out = ["## B1 Model against model", ""]
    out += [
        "### Single notes",
        "",
        "One 8' pipe each. Level in dB relative to full scale; attack is the time to 90 % of "
        "the steady level, release the time to fall 20 dB; \"between harmonics\" is the share "
        "of the power that is not in the harmonics (breath noise, aliasing); H2 to H8 are the "
        "levels of the partials relative to the first.",
        "",
    ]
    headers = ["note", "synth", "pitch (cents)", "level (dB)", "attack (ms)", "overshoot (%)"]
    headers += ["release (ms)", "between harmonics (dB)"] + [f"H{k} (dB)" for k in range(2, 9)]
    rows = []
    for label, frequency in NOTES.items():
        notes = {name: measure_note(organ(name), frequency) for name in SYNTHS}
        for name, note in notes.items():
            rows.append(
                [label, name, f"{note.detuning:+.2f}", f"{level_db(note.level):.1f}"]
                + [f"{1000 * note.attack:.0f}", f"{100 * (note.overshoot - 1):.0f}"]
                + [f"{1000 * note.release:.0f}", f"{note.inharmonic:.1f}"]
                + _partials(note.partials)
            )
        modal, waveguide = notes["organ_modal"], notes["organ_waveguide"]
        diff = note_difference(modal, waveguide)
        rows.append(
            [label, "modal - waveguide", f"{diff.pitch:+.2f}", f"{diff.level:+.1f}"]
            + [f"{1000 * diff.attack:+.0f}", f"{100 * diff.overshoot:+.0f}"]
            + [f"{1000 * diff.release:+.0f}", f"{diff.inharmonic:+.1f}"]
            + _partials(diff.partials, signed=True)
        )
    out += table(headers, rows)

    out += ["### Whole piece", ""]
    files = {name: directory / f"{name}_dry.wav" for name in SYNTHS}
    if all(path.exists() for path in files.values()):
        out += [
            "Level in dB of the dry renderings of `examples/organ_comparison.py` in octave "
            "bands, relative to the strongest band of each file.",
            "",
        ]
        levels = {}
        for name, path in files.items():
            audio, sample_rate = read_wav(path)
            levels[name] = band_levels(audio, sample_rate)
            levels[name] -= levels[name].max()
        headers = ["synth"] + [f"{centre:.0f} Hz" for centre in OCTAVE_BANDS]
        rows = [[name] + [f"{level:.1f}" for level in levels[name]] for name in SYNTHS]
        difference = levels["organ_modal"] - levels["organ_waveguide"]
        rows.append(["modal - waveguide"] + [f"{level:+.1f}" for level in difference])
        out += table(headers, rows)
    else:
        out += [f"Left out: no `{files['organ_modal'].name}` etc. in `{directory}`.", ""]
    return out


def _partials(levels: tuple[float, ...], signed: bool = False) -> list[str]:
    """Partials 2 to 8 as table cells; a dash where the sample rate has no room for one."""
    form = "{:+.1f}" if signed else "{:.1f}"
    return [form.format(levels[k]) if k < len(levels) else "-" for k in range(1, 8)]


def physical_consistency() -> list[str]:
    out = ["## B2 Physical consistency", ""]
    out += [
        "### Resonances against theory",
        "",
        "The resonances of an unblown pipe, measured from how it rings when tapped, against "
        "(a) the detailed theory - end corrections that shrink with frequency, losses to "
        "walls, radiation and mouth; the modal synth is built from these formulas - and "
        "(b) the ideal textbook pipe, a cylinder open at both ends without a mouth, whose "
        "resonances are exact harmonics (a dash where its low-frequency radiation formula no "
        "longer holds). Frequencies in cents above the reference, decay times as a ratio to "
        "the reference.",
        "",
    ]
    headers = ["pipe", "synth", "resonance", "frequency (Hz)", "vs detailed (cents)"]
    headers += ["vs ideal (cents)", "decay (ms)", "/ detailed", "/ ideal"]
    rows = []
    for label, frequency in RESONANCE_NOTES.items():
        for name, pipe in SYNTHS.items():
            measured = measure_resonances(
                pipe(frequency, SAMPLE_RATE), SAMPLE_RATE, frequency, RESONANCES
            )
            detailed = resonance_deviations(measured, detailed_resonances(frequency, RESONANCES))
            ideal = resonance_deviations(measured, ideal_resonances(frequency, RESONANCES))
            detailed = {deviation.number: deviation for deviation in detailed}
            ideal = {deviation.number: deviation for deviation in ideal}
            for resonance in measured:
                n = resonance.number
                rows.append(
                    [label, name, n, f"{resonance.frequency:.2f}"]
                    + [f"{detailed[n].cents:+.1f}" if n in detailed else "-"]
                    + [f"{ideal[n].cents:+.1f}" if n in ideal else "-"]
                    + [f"{1000 * resonance.decay_time:.2f}"]
                    + [f"{detailed[n].decay_ratio:.2f}" if n in detailed else "-"]
                    + [f"{ideal[n].decay_ratio:.2f}" if n in ideal else "-"]
                )
    out += table(headers, rows)

    notes = {name: measure_notes(organ(name), COMPASS) for name in SYNTHS}
    out += [
        "### Tuning",
        "",
        f"Error of the sounding pitch of all {len(COMPASS)} pipes from {COMPASS[0]:.1f} Hz to "
        f"{COMPASS[-1]:.0f} Hz, a semitone apart, with the random mistuning of the pipes "
        "switched off.",
        "",
    ]
    rows = []
    for name in SYNTHS:
        errors = np.array([note.detuning for note in notes[name]])
        worst = int(np.abs(errors).argmax())
        rows.append(
            [name, f"{np.abs(errors).mean():.3f}", f"{errors[worst]:+.3f}", f"{COMPASS[worst]:.1f}"]
        )
    out += table(["synth", "mean error (cents)", "worst error (cents)", "at (Hz)"], rows)

    out += [
        "### Scaling over the compass",
        "",
        "Power laws `value ~ frequency ^ exponent` fitted to the same pipes. The level is "
        "regulated to an exponent of -0.25; a time that is a fixed number of periods at every "
        "pitch has -1. \"Worst\" is how far a single pipe strays from the law: ranks should "
        "change smoothly from bass to treble.",
        "",
    ]
    quantities = {
        "level": lambda note: note.level,
        "attack time": lambda note: note.attack,
        "release time": lambda note: note.release,
        "2nd partial": lambda note: 10 ** (note.partials[1] / 20),
        "4th partial": lambda note: 10 ** (note.partials[3] / 20),
    }
    rows = []
    for quantity, value in quantities.items():
        for name in SYNTHS:
            pairs = []
            for frequency, note in zip(COMPASS, notes[name]):
                try:
                    pairs.append((frequency, value(note)))
                except IndexError:  # no such partial below half the sample rate
                    pass
            pairs = [pair for pair in pairs if pair[1] > 0 and math.isfinite(pair[1])]
            law = power_law(*zip(*pairs))
            rows.append([quantity, name, len(pairs), f"{law.exponent:+.3f}", f"{law.worst:.2f}"])
    out += table(["quantity", "synth", "pipes", "exponent", "worst (dB)"], rows)
    return out


def _aliasing(job: tuple[str, float]) -> tuple[float, list[tuple[float, float]]]:
    """Aliasing of one pipe without breath noise, and the tones that stand out with it."""
    name, frequency = job
    level = aliasing_level(organ(name, voicing=SILENT_JET), frequency)
    return level, inharmonic_tones(organ(name), frequency)


def numerical_quality() -> list[str]:
    out = ["## B3 Numerical quality", ""]
    out += [
        "### Aliasing",
        "",
        f"Worst case per octave over all {len(COMPASS)} pipes. \"Aliasing\" is the share of "
        "the power between the harmonics of a steady note when the jet makes no breath noise, "
        "i.e. what the simulation adds by mistake; the synths oversample their higher pipes "
        f"to keep it below {ALIASING_TARGET} dB. \"Tones\" counts the pipes in which, with "
        "breath noise as in normal operation, a tone stands out by 10 dB from the noise "
        "between the harmonics; the strongest is given in dB relative to its fundamental.",
        "",
    ]
    jobs = [(name, frequency) for name in SYNTHS for frequency in COMPASS]
    with ProcessPoolExecutor() as pool:  # one pipe per job: independent of each other
        measured = dict(zip(jobs, pool.map(_aliasing, jobs)))
    rows = []
    for octave in range(math.floor(math.log2(COMPASS[-1] / COMPASS[0])) + 1):
        pipes = [f for f in COMPASS if math.floor(math.log2(f / COMPASS[0]) + 1e-9) == octave]
        for name in SYNTHS:
            levels = [measured[name, frequency][0] for frequency in pipes]
            tones = [tone for frequency in pipes for tone in measured[name, frequency][1][:1]]
            factors = sorted({organ(name).oversampling(frequency) for frequency in pipes})
            worst = int(np.argmax(levels))
            strongest = max(tones, key=lambda tone: tone[1], default=None)
            rows.append(
                [f"C{octave + 1} up", name, len(pipes)]
                + [f"{factors[0]}x" if len(factors) == 1 else f"{factors[0]}x to {factors[-1]}x"]
                + [f"{levels[worst]:.1f}", f"{pipes[worst]:.0f}", len(tones)]
                + [f"{strongest[1]:.1f} dB at {strongest[0]:.0f} Hz" if strongest else "-"]
            )
    headers = ["octave", "synth", "pipes", "oversampling", "aliasing (dB)", "at (Hz)"]
    out += table(headers + ["tones", "strongest"], rows)
    overall = max(level for level, _ in measured.values())
    verdict = "met" if overall < ALIASING_TARGET else "NOT met"
    out += [f"Worst of all: {overall:.1f} dB; the target of {ALIASING_TARGET} dB is {verdict}.", ""]

    out += [
        "### Convergence with the sample rate",
        "",
        f"The same note at {SAMPLE_RATES[0]} Hz and {SAMPLE_RATES[1]} Hz, as a difference "
        f"from {SAMPLE_RATES[-1]} Hz. \"Partials\" is the largest difference among partials "
        "2 to 6.",
        "",
    ]
    rows = []
    for label, frequency in RESONANCE_NOTES.items():
        for name in SYNTHS:
            differences = convergence(lambda rate: organ(name, rate), frequency, SAMPLE_RATES)
            for rate, diff in zip(SAMPLE_RATES, differences):
                partials = max(abs(level) for level in diff.partials[1:6])
                rows.append(
                    [label, name, rate, f"{diff.pitch:+.3f}", f"{diff.level:+.2f}"]
                    + [f"{partials:.2f}", f"{1000 * diff.attack:+.1f}"]
                    + [f"{diff.inharmonic:+.1f}"]
                )
    headers = ["note", "synth", "sample rate (Hz)", "pitch (cents)", "level (dB)"]
    headers += ["partials (dB)", "attack (ms)", "between harmonics (dB)"]
    out += table(headers, rows)

    out += [
        "### Stability",
        "",
        f"Whether each of the {len(COMPASS)} pipes of the compass speaks its fundamental, "
        "holds a steady level, stays below full scale and dies away when released.",
        "",
    ]
    rows = []
    for name in SYNTHS:
        checks = stability(organ(name), COMPASS)
        failed = [f"{check.frequency:.1f} Hz: {check.problem}" for check in checks if not check.ok]
        steadiest = max(check.fluctuation for check in checks if check.ok)
        rows.append(
            [name, len(checks), len(checks) - len(failed), f"{100 * steadiest:.1f}"]
            + ["; ".join(failed) or "-"]
        )
    headers = ["synth", "pipes", "pass", "largest level variation (%)", "failures"]
    out += table(headers, rows)
    return out


def cost() -> list[str]:
    out = ["## B5 Cost", ""]
    out += [
        "Three seconds of a note. The real-time factor is seconds of audio per second of "
        "computing on one core; \"voicing\" is the time spent once per pipe and process on "
        "tuning and regulating it. High pipes are oversampled and cost that many times more. "
        "These numbers depend on the computer and vary from run to run.",
        "",
    ]
    playing = [
        ("A3, one pipe", 220.0, (8,)),
        ("A3, chorus 8' + 4' + 2'", 220.0, (8, 4, 2)),
        ("A6, one pipe", 1760.0, (8,)),
    ]
    rows = []
    for name in SYNTHS:
        for label, frequency, stops in playing:
            PipeOrganSynth.forget_tunings()
            segments = [{"duration": 3.0, "frequency": frequency, "stops": stops}]
            result = measure_cost(lambda: create_synth(name, sample_rate=SAMPLE_RATE), segments)
            factors = [f"{organ(name).oversampling(frequency * 8 / stop)}x" for stop in stops]
            rows.append(
                [name, label, " + ".join(factors), f"{result.realtime_factor:.1f}"]
                + [f"{result.preparation:.2f}"]
            )
    headers = ["synth", "playing", "oversampling", "real-time factor", "voicing (s)"]
    out += table(headers, rows)
    return out


def main(directory: str = "output") -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    report = [
        "# Metrics of the organ synths",
        "",
        f"Written by `examples/organ_metrics.py` at a sample rate of {SAMPLE_RATE} Hz. No "
        "recordings of a real organ are involved: the numbers say how the two synths differ, "
        "whether they behave like pipes and whether the numerics are sound, not how real "
        "they sound.",
        "",
    ]
    for section in (lambda: model_against_model(out), physical_consistency, numerical_quality):
        lines = section()
        print("\n".join(lines), flush=True)
        report += lines
    lines = cost()
    print("\n".join(lines))
    report += lines
    path = out / "organ_metrics.md"
    path.write_text("\n".join(report))
    print(f"written to {path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
