# GitHub Presentation Checklist

These settings make NumPyForge easier to discover and evaluate. The repository
metadata (description + topics) is applied live via the GitHub API; the social
preview image must be uploaded once through the web UI.

## About Section

- [x] **Description** (applied):

  ```text
  Built a pure-NumPy ML framework with classical models, MLPs, evaluation tools, FastAPI serving, Docker packaging, versioned artifacts, and CI-enforced tests/type checks.
  ```

- [x] **Topics** (applied):

  ```text
  machine-learning
  numpy
  ml-from-scratch
  mlops
  fastapi
  docker
  github-actions
  gradient-descent
  neural-networks
  model-serving
  ```

  Reapply with the GitHub CLI:

  ```bash
  gh repo edit --add-topic machine-learning,numpy,ml-from-scratch,mlops,fastapi,docker,github-actions,gradient-descent,neural-networks,model-serving
  ```

- Website (optional): the CI workflow or a hosted demo, e.g.
  `https://github.com/yashasviudayan-py/NumPyForge/actions/workflows/ci.yml`

## Social Preview

GitHub social previews must be a raster image (PNG/JPG, 1280x640). The card is
authored as `assets/social-preview.svg` and rendered to
`assets/social-preview.png` (the file to upload).

- [ ] **Upload** (manual, web UI — GitHub has no API for this):
  `Settings -> General -> Social preview -> Edit -> Upload an image`, then pick
  `assets/social-preview.png`.

Re-render the PNG from the SVG after any edit:

```bash
rsvg-convert -w 1280 -h 640 assets/social-preview.svg -o assets/social-preview.png
```

> `rsvg-convert` ships with `librsvg` (`brew install librsvg`). It preserves the
> 1280x640 aspect ratio; macOS `qlmanage` does not and will clip the card.

## Pinned Repo Blurb

```text
Built a NumPy-only ML framework and production pipeline: classical models, MLPs, metrics, artifacts,
FastAPI serving, Docker, and CI.
```
