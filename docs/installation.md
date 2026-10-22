# Installation

The project page and the segmentation model have separate requirements.

## View the project page

The hosted page is https://oxidationreduction.github.io/UniSEM-ISBI/.
To use the same files locally:

Python 3.10 or newer is sufficient. No installation, model weights, network
connection, or GPU is needed for the bundled prediction viewer:

```bash
python tools/serve.py
```

Open **http://127.0.0.1:8787**. Press Ctrl+C in the server terminal to stop it.
Use `--port 8788` if another application occupies the default port.
The server exposes only `site/`, not local model weights or datasets.

## Install the model package

Use Linux and Python 3.10 for the reference ML environment. On Windows, use
WSL2 or a Linux machine for the model pipeline; the static viewer runs natively.
VIGRA, Nifty, and Affogato are native scientific dependencies, so a plain
Windows `pip install` is not the reference installation.

```bash
conda env create -f environment.sam.yml
conda activate unisem
python -m pip install --constraint constraints.txt -e '.[test]'
unisem --help
python -m pytest -q
```

The pinned Python packages match the reference PyTorch 2.6 / micro-SAM 1.7.5
environment. Select the PyTorch CPU or CUDA build for your platform before the
editable install if necessary. No CUDA device is required for model inspection,
evaluation, graph processing, or the website. ViT-L dense inference is supported
on CPU but is substantially more practical on a CUDA device.

The model package uses published upstream dependencies. It does not import
private experiment controllers or require a particular server directory.
See [validation](validation.md) for the environment and checks actually run
for this release; an installed environment and a fresh dependency solve are
different verification levels.
