# Preference conditioning on a real model — measured

Everything else in this repo is synthetic. This is the one result measured on a real
instruction-tuned model: **Qwen/Qwen3-0.6B**, downloaded weights, greedy decoding, no
training. It tests the premise the whole method rests on — that a policy can be steered by
a preference direction placed in its prompt.

Task: two-digit multiplication (`47 * 83`), chosen because the model is at ~75% accuracy
on it rather than at ceiling, so there is accuracy available to trade away. 8 problems per
direction, 5 directions across the simplex.

## The result

| conditioning | controllability (accuracy) | controllability (brevity) | mean |
|---|---|---|---|
| **numeric** — `<preference>accuracy=0.70 brevity=0.30</preference>` | +0.000 | +0.000 | **+0.000** |
| **verbal** — instructions rendered from the same weights | +0.821 | +0.900 | **+0.860** |

Same model, same task, same weight vectors. Only the encoding changed.

Under numeric conditioning the model produced **identical** behaviour at every point on
the simplex — accuracy 0.750 and brevity 0.753 at all five directions, 8 words every time.
Under verbal conditioning it traces a real frontier:

| requested `w` | accuracy | brevity |
|---|---|---|
| `[0.00, 1.00]` | 0.625 | **0.817** |
| `[0.25, 0.75]` | 0.750 | 0.777 |
| `[0.50, 0.50]` | 0.875 | 0.646 |
| `[0.75, 0.25]` | 0.750 | 0.763 |
| `[1.00, 0.00]` | **1.000** | 0.202 |

The model was never the problem. A separate check confirms it is *highly* steerable — plain
instructions swing it from 3 words to 68, and from 0.500 to 0.750 accuracy:

| instruction | accuracy | brevity | mean words |
|---|---|---|---|
| `<preference>accuracy=1.00 brevity=0.00</preference>` | 0.750 | 0.690 | 8 |
| `<preference>accuracy=0.00 brevity=1.00</preference>` | 0.750 | 0.685 | 8 |
| "Show every step of your working in detail." | 0.750 | 0.123 | 68 |
| "Answer with the number only. No working." | 0.500 | 0.828 | 3 |

An instruction-tuned model has seen enormous amounts of text telling it how to behave and
essentially none pairing a decimal weight vector with a behaviour. The numeric format asks
it to do something it was never trained to do.

## Why this would have been expensive to discover later

`DESIGN.md` §5 chose prompt conditioning as the default and named its weakness — nothing
structurally forces the model to attend to the control line. That concern was correct, and
the numeric encoding fails it completely.

The consequence is specific to PaCE. Its cross-direction advantage matrix requires the `K`
rollouts in a group to actually differ by direction. If the policy ignores `w` at
initialization, every rollout in the group is identical, `S` is rank-one, and the mechanism
has nothing to bootstrap from — the exact degenerate case
`tests/test_advantages.py::test_homogeneous_directions_collapse_the_mechanism` asserts
against. A run would not have crashed. It would have trained, produced a plausible loss
curve, and quietly optimized a single operating point while reporting a "frontier" that was
one repeated behaviour.

Verbal conditioning is now the default (`build_prompt(..., style="verbal")`);
`style="numeric"` is retained for ablations and for models fine-tuned to read weights.

## A reward bug found the same way

Generating from a real model also exposed a bug in `extract_answer`. Qwen3-0.6B answers
`What is 37+58?` with `#### 37 + 58 = 95` — marker first, answer last. The old regex
captured the number immediately following `####`, returning **37**, so a fully correct
completion scored zero accuracy.

Nothing in a training curve would have revealed this; the accuracy objective would simply
have looked impossibly hard. Fixed to take the last number after the final marker, which is
correct under both that convention and GSM8K's `reasoning...\n#### 95`. Regression test:
`test_extract_answer_handles_marker_first_completions`.

## Caveats

- One model, one task family, 8 problems per direction. The effect is large enough to be
  unambiguous but the numbers are not precise.
- Zero-shot only. RL could in principle teach a model to read numeric weights; the point is
  that it would start from no signal, which is a much worse place to begin than +0.860.
- The bands in `verbalize_preference` are coarse by design — three levels per objective.
  Phrasing is what the model responds to, and interpolating text does not produce
  interpolated behaviour. Frontier resolution comes from which objectives are emphasized,
  not from fine gradations in how emphatically.
