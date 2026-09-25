"""Score a dictionary's reads against the toy generator's ground truth; synthdict is the caller.

`core` holds the detectors, gates, pair frame and census classifier; `benchmark` grades one scored
read under the shared `metrics.rules` rules and writes its artifacts; `config` holds the settings
only this pipeline uses.
"""
