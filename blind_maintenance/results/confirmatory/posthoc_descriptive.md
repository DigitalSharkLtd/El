# Post-hoc descriptive notes (NOT in the frozen plan)

Computed after the confirmatory run. Descriptive only; no intervals, no tests.

## Deaths in dynamic worlds: count and hidden regime at the death step

| strategy | deaths | regime at death slow/mid/fast | worlds with final S > 0.4532 (y=1 for ever) |
|---|---|---|---|
| fixed_5 | 14 | 0/0/14 | 54 of 512 |
| output_trend | 59 | 2/0/57 | 175 of 512 |
| C | 27 | 0/0/27 | 104 of 512 |
| C_level | 29 | 0/0/29 | 108 of 512 |
| C_scrambled | 31 | 0/0/31 | 107 of 512 |
| oracle_D | 16 | 0/0/16 | 62 of 512 |

Worlds where both C and fixed_5 died: 14; only C: 13; only fixed_5: 0.

## What triggered C's SERVICE requests (step logs of the 6 logged worlds)

| mode | strategy | living steps | requests | via reactive E<0.965 | via risk>0.24 only | executed SERVICE |
|---|---|---|---|---|---|---|
| dynamic | C | 3000 | 2443 | 122 | 2321 | 473 |
| dynamic | C_level | 3000 | 2374 | 143 | 2231 | 471 |
| dynamic | C_scrambled | 3000 | 2889 | 148 | 2741 | 492 |
| static | C | 3000 | 690 | 94 | 596 | 323 |
| static | C_level | 3000 | 546 | 83 | 463 | 307 |
| static | C_scrambled | 3000 | 2868 | 106 | 2762 | 489 |

