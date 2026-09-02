# Morphological grammar engines — mostly untrained, scored on synthetic damage

26 engines scored by the scripts in `../benchmarks_semi_synthetic/`. Two
compounding reasons not to read their numbers as results:

1. **The substrate is synthetic.** See that directory's README — the benchmarks
   cut intact skeletons in software, so both halves of a "split" still share
   geometry that real v117 fragments do not.
2. **25 of the 26 contain no checkpoint-loading code at all.** They were scored
   at initialization. A score for an untrained model on a synthetic task is a
   measurement of the task, not of the method.

This was verified by reading the files rather than inferred from the directory's
history — see `docs/threads/experiment_survey.md`.

The architectures remain worth reading. The scores do not.
