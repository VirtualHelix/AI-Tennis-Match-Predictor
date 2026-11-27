# Deployment status

The target host is Vercel. This repository update organizes and tests the existing CourtIQ application; it does not yet represent a live Vercel deployment.

## Required artifacts

The actual model is intentionally not committed to Git. Obtain its export using `scripts/export_model.py` in the owner's working virtual environment. The export includes exact model dependency versions and the matching training source. Verify the manifest, model-source hash, Linux/Python compatibility, startup time, and peak memory before deployment.

`requirements/runtime.txt` contains the HTTP server dependencies. The production model dependencies must come from the verified export, rather than from the synthetic test environment in `requirements/dev.txt`.

The calibration CSV belongs beside `model.joblib`. If it is absent, the calibration chart is unavailable; metrics must not be reconstructed from an unrelated experiment.

## Server behavior already available

The shared service exposes predictions, streamed calculation milestones, player profiles, head-to-head, recent results, evaluation, and model methodology. The FastAPI adapter adds `/healthz`, bounded inference concurrency, JSON request limits, and exact public-origin checks.

The adapter can be run as a conventional single-process server after the model and environment have been verified:

```bash
python -m uvicorn courtiq.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Set `COURTIQ_PUBLIC_ORIGIN` to the exact origin that serves the application. `/healthz` returns 503 when the model is unavailable. A healthy interface preview is not evidence of working model inference.

## Vercel work still required

The local results provider currently starts a daemon thread and stores its records, progress, rate limits, and daily quota in a local cache. The production Vercel implementation needs bounded refresh work and shared durable state with atomic coordination across function instances. Moving that cache to `/tmp` alone does not preserve it across instances or restarts.

Keep the existing score normalization, history reconciliation, and model math when introducing the durable results store. Add the Vercel entrypoint and configuration only after this behavior and the actual model bundle have been verified.

The release sequence is:

1. Push the reviewed source and tests to the existing GitHub repository.
2. Verify the real model bundle and finish the Vercel results-store integration.
3. Build and test the hosted app, including prediction streaming and updated results.
4. Publish the verified deployment URL and refresh the main README with the final architecture, setup, screenshots, and measured results.

No deployed URL or new accuracy result is claimed by this source update.

## Official platform references

- [Python applications on Vercel](https://vercel.com/docs/functions/runtimes/python)
- [FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi)
- [Function limits](https://vercel.com/docs/functions/limitations)
- [Hobby plan](https://vercel.com/docs/plans/hobby)
