"""Tokenization submodule.

We intentionally leave this `__init__.py` empty to prevent eager importing of
heavy dependencies (e.g., `datasets` and `transformers` via `tokenize.py` or
`tokenizer.py`). This allows lightweight modules like `config.py` to be imported
efficiently, ensuring fast startup times for our Hydra CLI applications.
"""
