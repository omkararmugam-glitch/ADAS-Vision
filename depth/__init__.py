"""Monocular depth -- MiDaS/DPT wrapper. PLACEHOLDER, build step 4.

ACCURACY WARNING, stated here because it governs how these outputs may be used:
MiDaS predicts *relative inverse* depth, not metres. It is defined only up to an
unknown scale and shift, so its raw output cannot yield a distance in metres
without being fitted to some metric reference.

This project already has a better metric reference for the common case: the
calibrated ground-plane homography in :mod:`core.calibrate` gives true metres for
any object whose contact point with the road is visible. The intended division of
labour is therefore:

  * **homography** -- primary, for objects touching the road (vehicles ahead,
    pedestrians). Accurate where the flat-ground assumption holds.
  * **MiDaS** -- fallback, for objects whose ground contact is occluded or
    off-plane, and as a consistency check that flags disagreement rather than
    silently trusting either source.

See ``docs/ARCHITECTURE.md`` for the full error analysis.
"""
