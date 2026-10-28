# BentoML

A learning project for training and serving a road accident Random Forest
classifier with BentoML. It includes basic prediction, service dependencies,
and JWT authentication examples.

## Setup

Run commands from the repository root. Install [uv](https://docs.astral.sh/uv/)
and use Python 3.12 for the pinned dependencies:

```bash
uv sync --python 3.12
```

This installs the locked dependencies into `.venv`. The current
`requires-python = ">=3.12"` declaration permits newer interpreters too,
but the pinned dependencies may not support them.

## Prepare data and train

Download the four road accident datasets for 2021, then preprocess them:

```bash
uv run python src/data/import_raw_data.py
uv run python src/data/make_dataset.py
```

When prompted by the preprocessing script, enter `data/raw` as the input
directory and `data/preprocessed` as the output directory. It writes
`X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv`.

Train the model:

```bash
uv run python src/models/train_model.py
```

The script prints the accuracy and saves the model as `accidents_rf` in the
local BentoML Model Store. The services load `accidents_rf:latest`, so train
before starting a service.

## Serve and predict

Start the JWT authentication example on port 3001:

```bash
make serve
```

In another terminal, obtain a token or send the example prediction request:

```bash
make token
make predict
```

These commands require `curl` and `jq`. The service exposes `/login` and
`/predict`. The example credentials are `user123` / `password123`;
the hardcoded credentials and JWT secret are for local learning only.

To explore the other services:

```bash
# Basic prediction service
make serve SERVICE=src.simple_service:RFClassifierService

# Separate API and model services using bentoml.depends
make serve SERVICE=src.simple_service_with_depends:RFClassifierService
```

Both alternatives expose `/predict` without `/login`. The Makefile's token
and prediction commands target the JWT example.
To change ports, use `PORT=3002` on both the serving and client commands.

## API smoke check

With the JWT service running on port 3001:

```bash
uv run python tests/test_api.py
```

This script sends login and prediction requests and prints their responses.
`make test-api` runs the Makefile's example prediction request.

## Project structure

```text
├── data/                               # Raw and preprocessed data
├── models/                             # Template model directory
├── notebooks/                          # Exploration notebooks
├── src/
│   ├── data/                           # Download and preprocessing
│   ├── features/                       # Feature engineering scaffolding
│   ├── models/                         # Training and legacy prediction
│   ├── visualization/                  # Visualization scaffolding
│   ├── simple_service.py               # Basic prediction service
│   ├── simple_service_with_depends.py   # API and model services
│   └── service.py                      # JWT authentication example
├── tests/test_api.py                   # HTTP smoke check
├── Makefile                            # Serving and client commands
├── pyproject.toml                      # Project dependencies
├── uv.lock                             # Locked dependency versions
└── LICENSE
```

`src/models/predict_model.py` is a legacy joblib example. It expects
`src/models/trained_model.joblib`, which the current training script does
not write. Use the BentoML services with the current training workflow.

## Origin

Based on the [DataScientest road accident template](https://github.com/DataScientest-Studio/Template_MLOps_accidents).
Maintained independently as [abulhawa/BentoML](https://github.com/abulhawa/BentoML).
See [LICENSE](LICENSE) for the original license and copyright notice.
