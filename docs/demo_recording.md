# Demo Recording Guide

Use this as the script for a 60-90 second GIF or screen recording. The repo is already set up so the
demo can be recorded without inventing a custom flow.

## Suggested Flow

1. Show the README top section with the badges and architecture diagram.
2. Run the portfolio demo:

   ```bash
   make demo
   ```

3. Point out the JSON summary fields:
   - `test_accuracy`
   - `artifact_dir`
   - API `ready_status`
   - sample prediction probabilities
4. Run the production smoke path:

   ```bash
   make ingest
   make train
   make evaluate
   ```

5. Open the GitHub Actions page and show the green `quality` and `docker-smoke` jobs.

## Voiceover

> NumPyForge starts with pure NumPy ML implementations and carries them into a production-style
> workflow. The demo trains a model, writes a versioned artifact, loads it through FastAPI, and checks
> health, readiness, metadata, and prediction endpoints. CI enforces formatting, linting, typing,
> coverage, pipeline smoke tests, and a Docker build.

## Optional Local Recording Commands

If `asciinema` is installed:

```bash
asciinema rec assets/numpyforge-demo.cast
make demo
exit
```

If `terminalizer` or a screen recorder is installed, record the same `make demo` flow and place the
exported GIF at:

```text
assets/numpyforge-demo.gif
```

Then add this line near the top of `README.md`:

```markdown
![NumPyForge demo](assets/numpyforge-demo.gif)
```
