# Project-page deployment

Repository: https://github.com/oxidationreduction/UniSEM-ISBI

Project page: https://oxidationreduction.github.io/UniSEM-ISBI/

## GitHub Pages

The repository uses the workflow in `.github/workflows/pages.yml`. In the
repository's **Settings → Pages**, select **GitHub Actions** as the publishing
source. A push to `main` or a manual workflow dispatch prepares and publishes
the existing `site/` directory.

The workflow uses Python's standard library to build the source-download ZIP,
synchronize model cards, and check source hygiene and local links. It does not
install PyTorch, train a model, run inference, or upload local model weights.
Website paths are relative so that the page works under `/UniSEM-ISBI/`.

The source download is generated as a Pages artifact. It is intentionally not
committed to Git. The project code is also available directly from GitHub.

After pushing, inspect the **Deploy project page** workflow in Actions. A
successful deployment provides the project-page URL. Check that the hosted
viewer, model cards, manuscript links, and source download work before treating
a deployment as complete. A local commit or push alone does not prove that the
site deployed successfully.

## Local preview

```bash
python tools/package_release.py
python tools/check_release.py
python tools/serve.py
```

Open http://127.0.0.1:8787. Press Ctrl+C in the server terminal to stop it.
The same static files power local preview and GitHub Pages.

## Model availability

First-stage weights and the paired source RF are publicly available on ModelScope:
https://modelscope.cn/models/dreamBuilder/UniSEM. The Colab notebook in this repository
uses immutable code and model revisions and verifies each file's SHA256.
The published model revision is `c074500d3426dfe4d43d47fb237775c2e2c788d5`.
Model files remain outside Git and the Pages artifact. Target checkpoints and
RFs remain local candidates, as recorded per asset in `models/manifest.json`.

After changing the notebook, test it in a clean Colab runtime before updating
its validation record. A browser-page preview or a CPU unit test alone does not
verify the Colab installation and real-model inference. Keep notebook outputs
and execution counts cleared in the committed file.
