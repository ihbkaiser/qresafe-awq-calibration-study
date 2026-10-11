# Public release manifest

This repository contains the complete recorded experiment package for the
Q-resafe AWQ calibration-data study, including source code, tests, Modal run
logs, checkpoints, calibration artifacts, evaluation summaries, and result
tables.

The raw artifacts are included because this is an explicitly requested full
public release. They may contain model-generated responses and safety-test
records; users should handle them as research data and avoid redistributing
harmful prompt content unnecessarily.

Excluded by `.gitignore`: credentials and secrets, Python/test caches, and
local temporary files. Model weights are not included; they remain gated and
are downloaded from Hugging Face at runtime using a Modal secret.

Main recorded runs:

- `remote_artifacts/r6_final/`: initial five-condition experiment.
- `remote_artifacts/r6_tokenmatched/`: token-matched Risk-I/Risk-II/Risk-III confirmation.
- `remote_artifacts/r6_checkpoints/`: raw resumable evaluation checkpoints.
- `remote_artifacts/*/`: Modal smoke, preparation, access-check, and full-run logs.

The public repository is not a claim that Risk-II is the unreleased original
Q-resafe dataset. Risk-II is reconstructed from the published description.
