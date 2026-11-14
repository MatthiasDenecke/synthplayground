# Metrics of the organ synths

Written by `examples/organ_metrics.py` at a sample rate of 44100 Hz. No recordings of a real organ are involved: the numbers say how the two synths differ, whether they behave like pipes and whether the numerics are sound, not how real they sound.

## B1 Model against model

### Single notes

One 8' pipe each. Level in dB relative to full scale; attack is the time to 90 % of the steady level, release the time to fall 20 dB; "between harmonics" is the share of the power that is not in the harmonics (breath noise, aliasing); H2 to H8 are the levels of the partials relative to the first.

| note | synth             | pitch (cents) | level (dB) | attack (ms) | overshoot (%) | release (ms) | between harmonics (dB) | H2 (dB) | H3 (dB) | H4 (dB) | H5 (dB) | H6 (dB) | H7 (dB) | H8 (dB) |
| ---- | ----------------- | ------------: | ---------: | ----------: | ------------: | -----------: | ---------------------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: |
| A2   | organ_waveguide   |         -0.00 |      -18.9 |         151 |             4 |          150 |                  -44.6 |    -7.7 |   -11.2 |   -19.6 |   -18.0 |   -26.5 |   -27.7 |   -34.9 |
| A2   | organ_modal       |         -0.00 |      -18.9 |         137 |             7 |          165 |                  -49.2 |    -8.1 |   -11.4 |   -20.2 |   -18.5 |   -27.7 |   -28.4 |   -36.8 |
| A2   | modal - waveguide |         -0.00 |       -0.0 |         -14 |            +2 |          +15 |                   -4.6 |    -0.4 |    -0.3 |    -0.6 |    -0.5 |    -1.2 |    -0.7 |    -1.9 |
| A3   | organ_waveguide   |         -0.01 |      -20.4 |          62 |             9 |           64 |                  -42.5 |    -7.6 |   -12.1 |   -20.8 |   -20.0 |   -28.3 |   -30.3 |   -36.9 |
| A3   | organ_modal       |         -0.01 |      -20.4 |          55 |            12 |           71 |                  -47.6 |    -8.1 |   -12.3 |   -21.5 |   -20.8 |   -30.2 |   -32.1 |   -40.9 |
| A3   | modal - waveguide |         -0.00 |       -0.0 |          -7 |            +3 |           +7 |                   -5.1 |    -0.5 |    -0.2 |    -0.7 |    -0.7 |    -1.9 |    -1.8 |    -3.9 |
| A4   | organ_waveguide   |         -0.01 |      -22.0 |          25 |            17 |           25 |                  -39.6 |    -7.6 |   -13.2 |   -22.4 |   -22.2 |   -30.1 |   -32.7 |   -38.6 |
| A4   | organ_modal       |         -0.01 |      -22.0 |          23 |            20 |           30 |                  -44.7 |    -8.3 |   -13.4 |   -23.3 |   -23.7 |   -33.6 |   -37.2 |   -47.0 |
| A4   | modal - waveguide |         +0.00 |       +0.0 |          -2 |            +3 |           +5 |                   -5.2 |    -0.7 |    -0.2 |    -1.0 |    -1.5 |    -3.6 |    -4.5 |    -8.4 |
| A5   | organ_waveguide   |         -0.01 |      -23.4 |          18 |             8 |           13 |                  -39.2 |    -7.5 |   -14.6 |   -24.1 |   -24.2 |   -31.5 |   -34.9 |   -39.9 |
| A5   | organ_modal       |         -0.01 |      -23.4 |          17 |            13 |           15 |                  -43.6 |    -8.4 |   -14.6 |   -25.3 |   -26.7 |   -37.6 |   -43.6 |   -57.7 |
| A5   | modal - waveguide |         +0.00 |       -0.0 |          -1 |            +5 |           +2 |                   -4.4 |    -0.9 |    -0.1 |    -1.3 |    -2.5 |    -6.1 |    -8.7 |   -17.8 |
| A6   | organ_waveguide   |         +0.00 |      -24.9 |          15 |             0 |            7 |                  -39.9 |    -7.4 |   -16.3 |   -26.2 |   -26.0 |   -32.3 |   -35.9 |   -39.9 |
| A6   | organ_modal       |         -0.02 |      -24.9 |          13 |             0 |            8 |                  -45.1 |    -8.6 |   -16.4 |   -28.2 |   -31.1 |   -44.3 |   -58.8 |   -54.6 |
| A6   | modal - waveguide |         -0.03 |       +0.0 |          -2 |            -0 |           +1 |                   -5.2 |    -1.2 |    -0.0 |    -2.1 |    -5.1 |   -12.0 |   -22.9 |   -14.7 |
| A7   | organ_waveguide   |         +0.02 |      -26.5 |          16 |             1 |            5 |                  -43.4 |    -7.3 |   -18.3 |   -28.8 |   -27.7 |       - |       - |       - |
| A7   | organ_modal       |         -0.01 |      -26.5 |          15 |             0 |            5 |                  -47.5 |    -9.0 |   -18.5 |   -32.0 |   -36.8 |       - |       - |       - |
| A7   | modal - waveguide |         -0.03 |       +0.0 |          -2 |            -0 |           +0 |                   -4.1 |    -1.6 |    -0.2 |    -3.3 |    -9.1 |       - |       - |       - |

### Whole piece

Level in dB of the dry renderings of `examples/organ_comparison.py` in octave bands, relative to the strongest band of each file.

| synth             | 62 Hz | 125 Hz | 250 Hz | 500 Hz | 1000 Hz | 2000 Hz | 4000 Hz | 8000 Hz | 16000 Hz |
| ----------------- | ----: | -----: | -----: | -----: | ------: | ------: | ------: | ------: | -------: |
| organ_waveguide   |  -7.4 |   -3.1 |    0.0 |   -1.7 |    -3.9 |    -9.1 |   -17.1 |   -27.6 |    -39.2 |
| organ_modal       |  -7.2 |   -3.1 |    0.0 |   -1.0 |    -4.1 |    -9.1 |   -17.7 |   -29.6 |    -45.3 |
| modal - waveguide |  +0.2 |   +0.0 |   +0.0 |   +0.6 |    -0.2 |    +0.0 |    -0.6 |    -2.0 |     -6.1 |

## B2 Physical consistency

### Resonances against theory

The resonances of an unblown pipe, measured from how it rings when tapped, against (a) the detailed theory - end corrections that shrink with frequency, losses to walls, radiation and mouth; the modal synth is built from these formulas - and (b) the ideal textbook pipe, a cylinder open at both ends without a mouth, whose resonances are exact harmonics (a dash where its low-frequency radiation formula no longer holds). Frequencies in cents above the reference, decay times as a ratio to the reference.

| pipe | synth           | resonance | frequency (Hz) | vs detailed (cents) | vs ideal (cents) | decay (ms) | / detailed | / ideal |
| ---- | --------------- | --------: | -------------: | ------------------: | ---------------: | ---------: | ---------: | ------: |
| A2   | organ_waveguide |         1 |         110.00 |                -0.1 |             -0.1 |     148.23 |       0.99 |    0.49 |
| A2   | organ_waveguide |         2 |         220.38 |                +0.6 |             +3.0 |      96.76 |       1.06 |    0.76 |
| A2   | organ_waveguide |         3 |         331.45 |                +1.6 |             +7.6 |      62.69 |       1.05 |    0.92 |
| A2   | organ_waveguide |         4 |         443.37 |                +2.9 |            +13.2 |      43.22 |       1.02 |    1.03 |
| A2   | organ_waveguide |         5 |         556.18 |                +4.2 |            +19.3 |      31.84 |       1.00 |    1.13 |
| A2   | organ_waveguide |         6 |         669.84 |                +5.5 |            +25.6 |      24.80 |       0.98 |    1.23 |
| A2   | organ_modal     |         1 |         110.00 |                +0.0 |             +0.0 |     149.97 |       1.00 |    0.49 |
| A2   | organ_modal     |         2 |         220.30 |                +0.0 |             +2.4 |      91.45 |       1.00 |    0.72 |
| A2   | organ_modal     |         3 |         331.14 |                +0.0 |             +5.9 |      59.73 |       1.00 |    0.88 |
| A2   | organ_modal     |         4 |         442.63 |                +0.0 |            +10.3 |      42.24 |       1.00 |    1.01 |
| A2   | organ_modal     |         5 |         554.83 |                +0.0 |            +15.1 |      31.87 |       1.00 |    1.13 |
| A2   | organ_modal     |         6 |         667.71 |                -0.0 |            +20.1 |      25.26 |       1.00 |    1.25 |
| A4   | organ_waveguide |         1 |         439.96 |                -0.2 |             -0.2 |      28.37 |       0.99 |    0.61 |
| A4   | organ_waveguide |         2 |         884.01 |                +1.6 |             +7.9 |      15.83 |       1.03 |    0.89 |
| A4   | organ_waveguide |         3 |        1334.66 |                +4.0 |            +19.1 |       9.54 |       1.01 |    1.04 |
| A4   | organ_waveguide |         4 |        1792.52 |                +6.6 |            +31.7 |       6.45 |       0.98 |    1.17 |
| A4   | organ_waveguide |         5 |        2256.94 |                +9.1 |            +44.2 |       4.79 |       0.97 |    1.31 |
| A4   | organ_waveguide |         6 |        2726.78 |               +11.3 |            +56.0 |       3.80 |       0.97 |    1.46 |
| A4   | organ_modal     |         1 |         440.00 |                -0.0 |             -0.0 |      28.79 |       1.00 |    0.62 |
| A4   | organ_modal     |         2 |         883.21 |                +0.0 |             +6.3 |      15.31 |       1.00 |    0.86 |
| A4   | organ_modal     |         3 |        1331.60 |                +0.0 |            +15.2 |       9.46 |       1.00 |    1.03 |
| A4   | organ_modal     |         4 |        1785.68 |                +0.0 |            +25.1 |       6.58 |       1.00 |    1.20 |
| A4   | organ_modal     |         5 |        2245.06 |                -0.0 |            +35.1 |       4.95 |       1.00 |    1.35 |
| A4   | organ_modal     |         6 |        2709.04 |                -0.0 |            +44.7 |       3.92 |       1.00 |    1.51 |
| A6   | organ_waveguide |         1 |        1759.47 |                -0.5 |             -0.5 |       5.05 |       0.98 |    0.73 |
| A6   | organ_waveguide |         2 |        3557.17 |                +2.1 |            +18.2 |       2.46 |       1.01 |    1.01 |
| A6   | organ_waveguide |         3 |        5408.50 |                +5.4 |            +41.6 |       1.46 |       0.99 |    1.20 |
| A6   | organ_waveguide |         4 |        7307.77 |                +8.2 |            +64.6 |       1.02 |       1.00 |    1.43 |
| A6   | organ_waveguide |         5 |        9241.63 |                +9.8 |                - |       0.80 |       1.04 |       - |
| A6   | organ_waveguide |         6 |       11197.75 |               +10.0 |                - |       0.67 |       1.10 |       - |
| A6   | organ_modal     |         1 |        1760.00 |                -0.0 |             -0.0 |       5.16 |       1.00 |    0.75 |
| A6   | organ_modal     |         2 |        3552.84 |                -0.0 |            +16.1 |       2.44 |       1.00 |    1.00 |
| A6   | organ_modal     |         3 |        5391.52 |                -0.0 |            +36.2 |       1.47 |       1.00 |    1.21 |
| A6   | organ_modal     |         4 |        7273.02 |                -0.0 |            +56.4 |       1.01 |       1.00 |    1.42 |
| A6   | organ_modal     |         5 |        9189.57 |                +0.0 |                - |       0.76 |       0.99 |       - |
| A6   | organ_modal     |         6 |       11134.30 |                +0.1 |                - |       0.60 |       0.99 |       - |

### Tuning

Error of the sounding pitch of all 89 pipes from 32.7 Hz to 5274 Hz, a semitone apart, with the random mistuning of the pipes switched off.

| synth           | mean error (cents) | worst error (cents) | at (Hz) |
| --------------- | -----------------: | ------------------: | ------: |
| organ_waveguide |              0.008 |              +0.058 |  4434.9 |
| organ_modal     |              0.006 |              -0.024 |  3322.4 |

### Scaling over the compass

Power laws `value ~ frequency ^ exponent` fitted to the same pipes. The level is regulated to an exponent of -0.25; a time that is a fixed number of periods at every pitch has -1. "Worst" is how far a single pipe strays from the law: ranks should change smoothly from bass to treble.

| quantity     | synth           | pipes | exponent | worst (dB) |
| ------------ | --------------- | ----: | -------: | ---------: |
| level        | organ_waveguide |    89 |   -0.251 |       0.14 |
| level        | organ_modal     |    89 |   -0.250 |       0.03 |
| attack time  | organ_waveguide |    89 |   -0.777 |       6.31 |
| attack time  | organ_modal     |    89 |   -0.799 |       6.11 |
| release time | organ_waveguide |    89 |   -1.041 |       5.12 |
| release time | organ_modal     |    89 |   -1.044 |       4.57 |
| 2nd partial  | organ_waveguide |    89 |   +0.012 |       0.04 |
| 2nd partial  | organ_modal     |    89 |   -0.023 |       0.34 |
| 4th partial  | organ_waveguide |    87 |   -0.270 |       1.43 |
| 4th partial  | organ_modal     |    87 |   -0.335 |       2.73 |

## B3 Numerical quality

### Aliasing

Worst case per octave over all 89 pipes. "Aliasing" is the share of the power between the harmonics of a steady note when the jet makes no breath noise, i.e. what the simulation adds by mistake; the synths oversample their higher pipes to keep it below -70 dB. "Tones" counts the pipes in which, with breath noise as in normal operation, a tone stands out by 10 dB from the noise between the harmonics; the strongest is given in dB relative to its fundamental.

| octave | synth           | pipes | oversampling | aliasing (dB) | at (Hz) | tones | strongest |
| ------ | --------------- | ----: | ------------ | ------------: | ------: | ----: | --------: |
| C1 up  | organ_waveguide |    12 | 1x           |        -115.1 |      62 |     0 |         - |
| C1 up  | organ_modal     |    12 | 1x           |        -131.9 |      62 |     0 |         - |
| C2 up  | organ_waveguide |    12 | 1x           |        -101.2 |     123 |     0 |         - |
| C2 up  | organ_modal     |    12 | 1x           |        -114.4 |     123 |     0 |         - |
| C3 up  | organ_waveguide |    12 | 1x           |         -87.2 |     247 |     0 |         - |
| C3 up  | organ_modal     |    12 | 1x           |         -97.4 |     247 |     0 |         - |
| C4 up  | organ_waveguide |    12 | 1x to 2x     |         -74.2 |     466 |     0 |         - |
| C4 up  | organ_modal     |    12 | 1x           |         -79.8 |     494 |     0 |         - |
| C5 up  | organ_waveguide |    12 | 2x to 3x     |         -84.6 |     932 |     0 |         - |
| C5 up  | organ_modal     |    12 | 1x to 2x     |         -75.8 |     622 |     0 |         - |
| C6 up  | organ_waveguide |    12 | 3x to 5x     |         -88.2 |    1397 |     0 |         - |
| C6 up  | organ_modal     |    12 | 2x to 4x     |         -85.0 |    1109 |     0 |         - |
| C7 up  | organ_waveguide |    12 | 5x to 9x     |         -90.2 |    2349 |     0 |         - |
| C7 up  | organ_modal     |    12 | 4x to 7x     |         -87.5 |    3136 |     0 |         - |
| C8 up  | organ_waveguide |     5 | 9x to 11x    |         -92.8 |    4186 |     0 |         - |
| C8 up  | organ_modal     |     5 | 7x to 9x     |         -92.5 |    4186 |     0 |         - |

Worst of all: -74.2 dB; the target of -70 dB is met.

### Convergence with the sample rate

The same note at 44100 Hz and 88200 Hz, as a difference from 176400 Hz. "Partials" is the largest difference among partials 2 to 6.

| note | synth           | sample rate (Hz) | pitch (cents) | level (dB) | partials (dB) | attack (ms) | between harmonics (dB) |
| ---- | --------------- | ---------------: | ------------: | ---------: | ------------: | ----------: | ---------------------: |
| A2   | organ_waveguide |            44100 |        -0.003 |      +0.00 |          0.03 |        -0.4 |                   +0.2 |
| A2   | organ_waveguide |            88200 |        -0.004 |      -0.00 |          0.04 |        -0.1 |                   -0.3 |
| A2   | organ_modal     |            44100 |        -0.003 |      +0.00 |          0.20 |        +8.2 |                   +0.1 |
| A2   | organ_modal     |            88200 |        -0.004 |      -0.00 |          0.06 |        +1.8 |                   -0.4 |
| A4   | organ_waveguide |            44100 |        -0.008 |      -0.01 |          0.08 |        -0.0 |                   -0.1 |
| A4   | organ_waveguide |            88200 |        -0.001 |      +0.00 |          0.08 |        -0.0 |                   +0.5 |
| A4   | organ_modal     |            44100 |        -0.002 |      -0.00 |          0.92 |        +2.2 |                   +0.6 |
| A4   | organ_modal     |            88200 |        -0.002 |      +0.00 |          0.35 |        +1.3 |                   +1.2 |
| A6   | organ_waveguide |            44100 |        +0.003 |      -0.00 |          0.02 |        +1.6 |                   -6.2 |
| A6   | organ_waveguide |            88200 |        +0.001 |      -0.00 |          0.01 |        +0.9 |                   -2.2 |
| A6   | organ_modal     |            44100 |        -0.022 |      -0.00 |          1.07 |        +2.2 |                   -5.6 |
| A6   | organ_modal     |            88200 |        +0.001 |      +0.00 |          0.01 |        +1.0 |                   -0.6 |

### Stability

Whether each of the 89 pipes of the compass speaks its fundamental, holds a steady level, stays below full scale and dies away when released.

| synth           | pipes | pass | largest level variation (%) | failures |
| --------------- | ----: | ---: | --------------------------: | -------: |
| organ_waveguide |    89 |   89 |                         4.7 |        - |
| organ_modal     |    89 |   89 |                         1.4 |        - |

## B5 Cost

Three seconds of a note. The real-time factor is seconds of audio per second of computing on one core; "voicing" is the time spent once per pipe and process on tuning and regulating it. High pipes are oversampled and cost that many times more. These numbers depend on the computer and vary from run to run.

| synth           | playing                 | oversampling | real-time factor | voicing (s) |
| --------------- | ----------------------- | ------------ | ---------------: | ----------: |
| organ_waveguide | A3, one pipe            | 1x           |             33.4 |        0.05 |
| organ_waveguide | A3, chorus 8' + 4' + 2' | 1x + 1x + 2x |              8.7 |        0.09 |
| organ_waveguide | A6, one pipe            | 4x           |              7.8 |        0.02 |
| organ_modal     | A3, one pipe            | 1x           |              9.0 |        0.14 |
| organ_modal     | A3, chorus 8' + 4' + 2' | 1x + 1x + 2x |              2.2 |        0.24 |
| organ_modal     | A6, one pipe            | 3x           |              2.9 |        0.15 |
