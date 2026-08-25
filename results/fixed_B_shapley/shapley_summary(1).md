# TAE four-cell symmetric Shapley decomposition

CDC 2023 is treated as the temporal evaluation cohort.
The same frozen canonical model and score vector are used within each target.

Bootstrap budget levels are fixed absolute counts: for each target/q, B_early and B_all are defined from the observed point-estimate cohorts and held unchanged in all 500 bootstrap replicates.

Four cells:

- C00: early candidate set + early-derived absolute budget
- C10: all-record candidate set + early-derived absolute budget
- C01: early candidate set + all-record-derived absolute budget
- C11: all-record candidate set + all-record-derived absolute budget

Main symmetric attribution:

`total = Shapley availability + Shapley capacity`

Factorial identity:

`total = availability main + capacity main + interaction`

## Main results

| Outcome | q | Total | Shapley availability | Shapley capacity | Interaction |
|---|---:|---:|---:|---:|---:|
| Preterm delivery | 5% | +3.365 [+3.302, +3.426] | +1.157 [+1.092, +1.222] | +2.208 [+2.173, +2.246] | +0.335 [+0.264, +0.407] |
| Preterm delivery | 10% | +5.669 [+5.582, +5.766] | +2.149 [+2.056, +2.241] | +3.520 [+3.484, +3.568] | +0.512 [+0.436, +0.604] |
| NICU admission | 5% | +3.315 [+3.255, +3.378] | +1.191 [+1.131, +1.251] | +2.124 [+2.092, +2.156] | +0.294 [+0.226, +0.360] |
| NICU admission | 10% | +5.588 [+5.507, +5.674] | +2.114 [+2.031, +2.201] | +3.475 [+3.436, +3.521] | +0.415 [+0.341, +0.509] |
| Low birth weight | 5% | +3.964 [+3.885, +4.058] | +1.429 [+1.345, +1.523] | +2.534 [+2.496, +2.571] | +0.402 [+0.324, +0.487] |
| Low birth weight | 10% | +6.771 [+6.667, +6.891] | +2.601 [+2.491, +2.723] | +4.170 [+4.120, +4.221] | +0.661 [+0.553, +0.776] |

All values are percentage-point population event-capture contrasts with percentile 95% bootstrap intervals.

The bootstrap conditions on the two observed absolute budget levels. At the point estimate, the C00-to-C11 total coincides with the cohort-specific top-q endpoint contrast; bootstrap uncertainty is computed for those fixed factorial budget levels.

The previous ordered decomposition is retained only as an appendix quantity (`ordered_availability_at_all_budget` and `ordered_capacity_within_early_cohort`).
