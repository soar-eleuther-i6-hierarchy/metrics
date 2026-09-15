"""The frozen single-property benchmark: fixed expressions over the ten existing detectors.

This package MEASURES the metrics in `scoring.core`; it defines no new metric and changes none
of them. Its job is to evaluate the eight expressions registered in `PRECOMMIT.md` s4 on every
(toy, seed, read) and report what they do -- including where they fail, which is a deliverable
rather than a defect to repair.

Layout:
  `registry.py`    the eight frozen expressions and the frozen constants
  `predicates.py`  literal HIGH/LOW/NOT-HIGH/NOT-LOW/IN-BAND/SYM/FREQ-LOCAL/SURVIVES
  `calibrate.py`   the calibration/evaluation null split and the null quantiles
  `reads.py`       the oracle and trained reads, reduced to one shape
  `evaluate.py`    the four nested counts, distribution flags, and the four verdict labels
  `run_benchmark.py`  the CLI driver, the write guard, and provenance

Deliberately NOT reused: `scoring.trained.cascade.greedy_cascade`. Forward selection over
percentile levels x detectors x both tails is exactly the adaptive search a frozen benchmark
must not contain. It stays available as a separately reported comparator.
"""
