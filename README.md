# README

## Repository Status & Overview

This repository has been polished to accompany the research report available in
[`report/report.pdf`](report/report.pdf).

Please note the current state of the various components within the codebase:

- **Weave-of-Formal-Thought Fine-Tuning**: The sections of the repository
  dedicated to WoFT fine-tuning—as covered in the report—are in a polished form,
  though currently still in an alpha state.

- **Formal Engine**: While the core underlying components of the engine are
  cleanly written and provided for reference alongside the report, the engine's
  frontend ([`src/formal/engine/frontend.py`](src/formal/engine/frontend.py))
  has not yet been polished and remains in a preliminary, uncleaned state.

### Documentation & Experiments

Detailed guides on navigating the repository and running the experiments will be
released soon. In the meantime, the codebase serves as its own documentation,
and Hydra is fully integrated throughout the CLI tools to make configuration and
execution straightforward for users.

## Cloning & Submodules

This repository relies on a fork of the tree-sitter main repo as a submodule,
where the generate crate has been slightly customized to our needs and where we
have added a python binding that is used by the formal engine.

To clone the repository and initialize the submodule, use the standard two-step
cloning workflow:

```sh
git clone https://github.com/alexbouayad/formal.git
cd formal
git submodule update --init --recursive
```

## Shell Completions

Since the CLI scripts (`formal-ingest`, `formal-tokenize`, `formal-train`) are
powered by Hydra, they come with rich tab-completion out of the box.

If you want to enable tab-completion for your current terminal session, run the
following commands directly in your active terminal:

### Zsh

If you are using Zsh, you must first ensure that bash completion support is
enabled in your shell. Run this once per session (or add it to your `~/.zshrc`):

```sh
autoload -Uz compinit && compinit
autoload -Uz bashcompinit && bashcompinit
```

Then enable the script completions:

```sh
eval "$(formal-ingest -sc install=bash)"
eval "$(formal-tokenize -sc install=bash)"
eval "$(formal-train -sc install=bash)"
```

### Bash

```sh
eval "$(formal-ingest -sc install=bash)"
eval "$(formal-tokenize -sc install=bash)"
eval "$(formal-train -sc install=bash)"
```

### Fish

```sh
formal-ingest -sc install=fish | source
formal-tokenize -sc install=fish | source
formal-train -sc install=fish | source
```
