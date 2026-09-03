"""Rules-based decision engine over all module outputs. PLACEHOLDER, build step 5.

Consumes per-frame perception records and emits structured, timestamped alerts:
lane departure, forward collision (TTC below threshold), pedestrian-in-path,
red-light running, and road-hazard flags -- with thresholds adapted to the
classified weather, since stopping distance in rain is not stopping distance on
dry tarmac.

Rules rather than a learned policy, because every alert has to be explainable
after the fact and tunable without retraining. The rules are pure functions over
typed records, which is what makes them unit-testable with no video and no GPU.
"""
