# Run CourtIQ locally

CourtIQ is the browser application for the saved Research V5 tennis predictor. The existing `src/` tree and `run_predictor.py` remain available as the original command-line project.

## If CourtIQ already works on your computer

From the repository, activate the same virtual environment that currently loads your model and run:

```bash
source .venv/bin/activate
python run_courtiq.py --port 8001
```

Open `http://127.0.0.1:8001`. The launcher opens a browser automatically unless you add `--no-browser`.

The preferred model location is `models/courtiq/model.joblib`. Existing models at `models/research_v5/model.joblib` are found automatically. Both locations are ignored by Git. The exact model source is kept in `courtiq/model.py`; its bytes must match the source hash saved in the model. Moving the source into this package does not change the hash.

To use a different artifact or its exact training source:

```bash
python run_courtiq.py --port 8001 \
  --model /absolute/path/to/model.joblib \
  --model-code /absolute/path/to/tennis_research_model.py
```

Use the package versions from the environment that trained the saved artifact. Installing newer scikit-learn or NumPy versions is not a substitute for verifying model compatibility.

## Development and tests

The software-test environment uses Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/dev.txt
python -m unittest discover -s tests -v
```

The HTTP integration tests train a small synthetic model in a temporary directory. Those records test software behavior and are never presented as real tennis predictions. No synthetic model is included in the application.

The local launcher itself remains compatible with Python 3.9 when used with an existing compatible model environment. The FastAPI server dependencies require Python 3.10 or newer.

## Updated match results

Set `SPORTSTATS_KEY` in the process environment before starting CourtIQ. `.env.example` documents supported variables; the Python launcher does not automatically read `.env` files.

Updated results supply displayed match records, set scores, and head-to-head history. They do not refresh the trained model's features or change its weights. Missing per-set data is identified as unavailable; an aggregate set count is never invented into individual scores.

The local server performs results refresh in the background and preserves its cache outside the Git repository. Without an API key, the app uses the match history in the saved model.

## Export the deployment model

Run the helper in the environment that successfully loads your existing model:

```bash
python scripts/export_model.py
```

It creates `~/Downloads/courtiq_model_bundle.zip` containing the model, matching source, optional calibration CSV, exact installed model dependency versions, and an integrity manifest. Existing output files are preserved; choose a new `--out` filename for another export. The helper does not load, retrain, publish, or upload the model.

## Code map

| Location | Purpose |
| --- | --- |
| `run_courtiq.py` | Small local launcher |
| `courtiq/server.py` | Shared prediction service and loopback HTTP server |
| `courtiq/api.py` | FastAPI adapter and streamed prediction endpoints |
| `courtiq/model.py` | Exact research training and prediction source |
| `courtiq/sportstats.py` | Results provider, quota controls, and set-score parsing |
| `courtiq/history.py` | Recorded head-to-head and recent-match reconciliation |
| `courtiq/ui.py` | Frontend assembly with photographs, credits, and facts |
| `courtiq/static/index.html` | Editable website layout, styles, and browser behavior |
| `courtiq/assets/` | Player photographs, attribution, and sourced tennis facts |
| `tests/` | Model-serving, API, history, provider, and export tests |

Source and license information for player photographs remains in `courtiq/assets/players/attribution.json` and in the rendered website. Tennis facts retain their source URLs.
