# WebNeuro Investigation

## Current Issues

The current version of WebNeuro has a couple of data quality issues:

1. **Timeout vs. genuine failure is indistinguishable.** There is no way to tell whether a participant timed out on a task versus completed it but got everything wrong.
2. **Missingness is sometimes coded as 0 instead of missing.** This is inconsistent across variables within the same task. For example, when a participant's RT is missing because the task timed out, RT is correctly labelled as missing, but `vi_sco1` (number of stimuli completed) is labelled as `0` instead of missing. Similarly, on the N-Back task, RT can be missing while false-positive errors are labelled `0` (because the participant never pressed anything), which inflates normed scores.

## Solution

Streamline the definitions of "missing" and "timeout" so that both are applied consistently across all tasks, and distinguish **timeout with effort** (participant engaged but ran out of time) from **timeout without effort / missing** (participant did not engage at all).

## Task Order

Per `webneuro_data_dictionary.json` (`test_order`), the real task order includes one task the original list skipped, and mislabels one later task:

1. Motor Tapping
2. Choice Reaction Time *(missing from the original list — `chlrrtav`)*
3. Verbal Memory
4. Emotion Identification
5. Digit Span (Forward)
6. Verbal Interference / Stroop
7. Switching of Attention
8. Go/No-Go
9. Delayed Memory
10. Emotion Priming / Delayed Face Recognition *(originally mislabelled "Emotion Identification" again — it's actually the old-vs-new face recognition task, `dgtcnA`/`dgtcrtA`/etc., distinct from #4)*
11. N-Back Continuous Performance Test
12. Maze

## Validation Logic by Task

### 1. Maze

The task times out at 16 minutes (dictionary range max is 1,000,000 msec ≈ 16.7 min, consistent).

- `emzcompk` — Total time to complete the maze (msec). **Dictionary rule: `emzcompk > emzinitk`.** Missing on timeout.
- `emzinitk` — Path-learning time: time up to the end of the final trial containing at least one error, i.e., just before completing the maze correctly twice in a row. Missing on timeout.
- `emzerrk` — Total errors of all types across all trials. **Dictionary rule: `emzerrk > emzoverk`** (strictly greater, not just `>=`). Missing on timeout.
- `emzoverk` — Cumulative "overrun" errors (continuing to press the same direction instead of turning where required). Missing on timeout.
- `emztrlsk` — Number of trials taken to complete the maze correctly twice. **Dictionary minimum is 2** (`emztrlsk >= 2`, not strictly `> 2`). Missing on timeout.

**Edge cases to flag:**
- `emzerrk <= emzoverk` — violates the dictionary's stated relationship; overruns are a subset of all errors, so total errors should always exceed overrun errors alone.
- `emztrlsk < 2` — below the documented floor.
- `emzcompk <= emzinitk` — violates the stated ordering between total time and path-learning time.
- **Partial missingness** — since all five Maze variables share the same "missing if timeout" note, they should go missing *together*. Flag any record where some Maze variables are populated and others are missing, since that likely reflects the same "no engagement vs. timed-out-with-effort" bug described in Issue 2 above, just not yet fixed for this task.
- `emzerrk = 0` or `emzoverk = 0` — the dictionary's stated range technically allows 0, but per your domain knowledge this is not expected in practice; flag (don't auto-correct) for manual review rather than assuming timeout.

### 2. Go/No-Go (GNG)

There are 168 trials total: 126 green "press" stimuli + **42** red "don't press" stimuli (not 48 — the dictionary's `g2fpk` range is `0::42`, and `126 + 42 = 168`, matching `g2errk`'s range).

- `g2avrtk` — Average RT between a green stimulus and the spacebar press, across correctly performed trials only. Range `0::2700` msec (interval between stimuli). **Missing if none correct** (not "no response to any stimulus" — a subtly different condition than the error variables below).
- `g2sdrtk` — *(not previously listed)* Standard deviation of RT across correctly performed green trials. Same "missing if none correct" condition as `g2avrtk`.
- `g2errk` — Total errors (`g2fnk + g2fpk`), range `0::168`. Missing if no response was made to *any* stimulus.
- `g2fnk` — False negatives/omissions (green shown, no press), range `0::126`. Same missing condition as `g2errk`.
- `g2fpk` — False positives/commissions (red shown, pressed anyway), range `0::42`. Same missing condition as `g2errk`.

**Consistency check:** `g2errk = g2fnk + g2fpk`.

**Edge cases to flag:**
- `g2fpk > 42` or `g2fnk > 126` or `g2errk > 168` — out of range per dictionary.
- `g2avrtk` present while `g2fnk = 126` — impossible: if every green trial was missed, there are zero correct trials to average, so `g2avrtk` (and `g2sdrtk`) must be missing.
- `g2sdrtk` missing/present should always match `g2avrtk` missing/present, since both derive from the same set of correct trials.
- **The exact bug this task is prone to:** `g2fnk = 126` *and* `g2fpk = 0` together means the participant never pressed the spacebar at all (every green stimulus is necessarily a miss, and no red stimulus was ever falsely pressed because nothing was pressed). Per the dictionary's own note ("missing if no response to any stimuli"), this combination should be relabeled fully missing — not reported as `126`/`0`. This is the GNG analog of the N-Back bug called out in Issue 2.

### 3. Stroop (Verbal Interference)

**Confirmed:** the task timeout is 30 seconds. The dictionary's recorded range for `vcrtne`/`vcrtne2` (`0::20000` msec) is itself wrong/stale — treat `vcrtne`/`vcrtne2 > 30000` msec as the out-of-range threshold for QC, not the dictionary's `20000`. This is a data dictionary error to flag separately, not a task design question.

- `vcrtne` — Average RT for correct color-word *name* selections (ignoring font color). **Missing if none correct** (not "if no response").
- `vi_sco1` — Number of stimuli with the name correctly selected inside the time limit. Range `0::`, no negative values. **Missing if no response** — a different condition than `vcrtne`'s.
- `vcrtne2` — Average RT for correct *font-color* selections (ignoring the word). **Missing if none correct.**
- `vi_sco2` — Number of stimuli with the font color correctly selected inside the time limit. Range `0::`. **Missing if no response.**
- `vi_difrt` — *(not previously listed)* Difference in RT between the first and second trial types, range `-20000::20000`. **Missing if none correct.** (This range is likely affected by the same stale-dictionary issue as `vcrtne`/`vcrtne2` — if the true timeout is 30s, the true bound should be `-30000::30000`.)

**Why `vi_sco1 = 0` is not automatically a bug:** `vi_sco1` and `vcrtne` use two *different* missingness conditions — "no response at all" vs. "no correct response." That means two genuinely different scenarios both produce `vi_sco1 = 0` with `vcrtne` missing:
- **Timeout with effort:** participant responded to every stimulus but got zero correct → `vi_sco1 = 0` is correct (a real score), `vcrtne` is correctly missing (no correct RT exists to average).
- **Missing without effort:** participant made zero responses at all → `vi_sco1` should be missing per its own dictionary note, but the observed bug codes it as `0`.

These two cases are **indistinguishable from `vi_sco1` and `vcrtne` alone** — both look like `vi_sco1 = 0` + `vcrtne` missing. Disambiguating them requires a total-response-count field (e.g., total attempts/trials presented, if logged at the raw trial level) that isn't in the summary-level dictionary. Recommend pulling raw trial counts for Stroop before deciding how to relabel `vi_sco1 = 0` cases.

**Edge cases to flag:**
- `vcrtne`/`vcrtne2` present while `vi_sco1`/`vi_sco2` (respectively) = 0 — contradiction, since a nonzero correct-RT average implies at least one correct trial.
- `vi_difrt` present while either `vcrtne` or `vcrtne2` is missing — `vi_difrt` is derived from both and should be missing whenever either input is missing.

### 4. Digit Span (Forward)

- `digitot` — Total correct trials, out of 14. Range `0::14`. **Missing if no response to all sequences.**
- `digitsp` — Longest correctly recalled span. Dictionary range is `0;3::9` — this is **not** a continuous `0–9` range. The valid values are `{0} ∪ {3, 4, 5, 6, 7, 8, 9}`; **1 and 2 are not valid values** (the shortest testable span is 3 digits, so a nonzero score can't be below 3). Missing under the same condition as `digitot`.

**Consistency check:** `digitot` and `digitsp` share the *identical* missing condition ("no response to all sequences"), so they should be missing together, not just in the `digitsp → digitot` direction previously noted. Also, if `digitsp = 0`, `digitot` should be `0` (inferred from task logic, not stated explicitly in the dictionary — worth confirming empirically).

**Edge cases to flag:**
- `digitsp = 1` or `digitsp = 2` — invalid per the dictionary's range notation; likely a data or scoring error.
- `digitot` missing while `digitsp` is present, or vice versa — violates the shared missingness condition.
- `digitsp = 0` while `digitot > 0`, or `digitot = 0` while `digitsp > 0` — contradicts task logic.

### 5. N-Back Continuous Performance Test *(added — this is the task named in Issue 2's motivating example)*

- `wmacck` — Total errors (`wmfnk + wmfpk`), range `0::63`. Missing if no response to any stimulus.
- `wmfnk` — False misses/omissions, range `0::12`. Same missing condition as `wmacck`.
- `wmfpk` — False alarms/commissions, range `0::51`. Same missing condition as `wmacck`. (`12 + 51 = 63`, matching `wmacck`'s max.)
- `wmrtk` — Average RT for correctly identified targets, range `0::2700` msec. **Missing if none correct** — a different condition than the error variables.

**This is exactly the Issue 2 bug pattern:** `wmrtk` (missing if none correct) and `wmfpk`/`wmfnk`/`wmacck` (missing if no response to *any* stimulus) use different missingness rules, just like GNG. If a participant never presses anything, `wmfnk = 12` and `wmfpk = 0` necessarily — and per the dictionary's own note, that combination should be fully missing, not reported as `12`/`0`.

**Edge cases to flag:**
- `wmfnk = 12` and `wmfpk = 0` together — no-engagement case; should be relabeled missing, not scored.
- `wmrtk` present while `wmfnk = 12` — impossible, since zero targets were hit correctly.
- `wmacck != wmfnk + wmfpk` — arithmetic inconsistency.
