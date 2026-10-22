# Segment your data in Colab

[Open the UniSEM notebook](https://colab.research.google.com/github/oxidationreduction/UniSEM-ISBI/blob/main/notebooks/UniSEM_Colab.ipynb)

The notebook runs the first-stage EMNeuron generalist and its matched full55
random forest in your own Colab runtime. It is a real inference workflow; the
project website's example comparison continues to use precomputed predictions.
The first-stage model and matching RF are publicly available from
[ModelScope · dreamBuilder/UniSEM](https://modelscope.cn/models/dreamBuilder/UniSEM).
WaferEM LightFT and FullFT assets remain local release candidates.

## Four steps

1. **Prepare.** Select a GPU runtime and run the setup cell. It installs an
   isolated Python 3.10 environment, checks the fixed source revision and
   downloads the published model/RF pair from
   [ModelScope · dreamBuilder/UniSEM](https://modelscope.cn/models/dreamBuilder/UniSEM).
2. **Upload.** Supply a grayscale TIFF image, a ZYX TIFF stack, or an HDF5 file.
   Choose the HDF5 dataset key (default `main`) and inspect the shape and dtype.
   An optional existing WaferEM display crop provides a quick example.
3. **Run.** Process the selected slice (default) or the complete stack. Inference
   uses the existing `unisem infer` command and the published instance-generation
   configuration, with no training or changes to image resolution.
4. **Inspect and download.** Select a result slice, toggle instances/boundaries
   and adjust opacity. Download the result ZIP before the runtime ends.

Uploads stay in the user's Colab runtime; the notebook does not send images to
the project website, ModelScope repository, or a project-operated inference
server. Colab itself provides the hosted runtime.

## Input and output

- Supported input dimensions: YX grayscale image or ZYX grayscale stack. RGB
  TIFF metadata, empty arrays, non-finite values and unsupported dimensions are
  rejected. Ambiguous HDF5 shapes ending in 3 or 4 require explicit ZYX
  confirmation; the notebook does not convert RGB into a volume.
- Input pixels and dimensions are preserved. The existing inference code uses
  per-slice 1st/99th percentile intensity normalization internally.
- `instances.tif` contains the exact predicted labels as uint32. The HDF5
  includes dense predictions, watershed fragments and
  `final_merged_segmentation`. `preview.png` shows the first output slice side
  by side with its boundary overlay; it is a visualization, not a metric.
- `run.json` records the input dimensions, scope, selected source slice, code
  and model revisions, asset hashes, device and measured inference elapsed time.
  Stack labels are independent 2D instances; matching IDs across slices do not
  indicate 3D connectivity.

## Reproducible environment

The notebook kernel handles upload and display. Model execution uses
`environment.colab.yml` with Python 3.10, VIGRA, Nifty and Affogato, followed by
the repository's `constraints.txt` and package dependencies. The ELF dependency
comes from `constantinpape/elf`; the unrelated package with the same name on
PyPI is not substituted.

The notebook pins a 40-character source commit and a fixed ModelScope model revision.
It verifies the two published SHA256 values again immediately before loading.
Only the matched published random forest is accepted; arbitrary uploaded
joblib/pickle files are not an input option.

The first-stage model is an EM generalist. This notebook does not use the
WaferEM-specialized LightFT/FullFT weights, and its bundled display crop is not
new performance evidence. Consult the model card for training provenance,
attribution, intended use and model-weight terms.

## Errors

A failed dependency install or model download stops setup with its actual error.
An invalid input reports its shape, key or format issue before inference.
Segmentation uses the actual CLI return status; failure clears the current
result panel instead of showing a synthetic or previous success. If an existing
runtime contains a different code revision, start a fresh runtime for the
notebook version you opened.

If Colab reports that it cannot load the JavaScript needed for output, refresh
your sign-in and consult the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
Rich outputs use a separate Google output origin; the FAQ describes a hostname
cookie exception instead of enabling third-party cookies globally. This affects
the browser's upload/viewer controls, separately from model computation.

## Release verification

Input validation, result export and all notebook code cells can be checked
without a GPU using `tests/test_demo.py`. A clean Colab installation and a real
small-image inference are separate release checks; report their actual status
in the release validation record rather than inferring them from unit tests.
