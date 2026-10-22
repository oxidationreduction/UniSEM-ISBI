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

Source code and the static project page are separate from model-weight
distribution. The candidate checkpoints and RFs remain outside Git and the
Pages artifact. Their identities and local preparation status are recorded in
`models/manifest.json`; a successful website deployment does not create a
public weight download or change their licensing status.
