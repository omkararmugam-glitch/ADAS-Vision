"""Dataset adapters: convert a raw third-party dataset into this project's format.

One adapter per source dataset. Each reads whatever the dataset ships and writes
the unified layout documented in ``docs/DATASET_FORMAT.md``. Adapters are
write-once and never read at inference time, so a dataset's quirks stay contained
here instead of leaking into the training or evaluation code.
"""
