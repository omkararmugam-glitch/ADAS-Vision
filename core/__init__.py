"""Shared infrastructure: config, typed records, IO, calibration, cache, overlay.

Nothing in this package imports a model or torch. That is a deliberate constraint,
not an accident: it is what allows ``python run.py render`` to rebuild an
annotated video from the cache in an environment where no weights and no
deep-learning stack are present.
"""
