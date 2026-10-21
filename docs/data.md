# Data and input conventions

## Your own data

`unisem infer` accepts a grayscale 2D TIFF, a ZYX TIFF stack, or an HDF5 dataset.
Supply the HDF5 key explicitly with `--input-key`; the default is `main`.
Segmentation IDs are nonnegative integers, with zero reserved for background.
The model predicts each slice independently. A stack output does not imply
cross-slice identity tracking or validated 3D connectivity.

The dense channel order is fixed:

1. Foreground probability.
2. Center distance.
3. Boundary distance.

These channels are **not** the three affinity channels used by some external
neuron segmentation methods.

## EMNeuron

Obtain the dataset from the [official EMNeuron repository](https://huggingface.co/datasets/yanchaoz/EMNeuron).
The reference split uses 79 training pairs, nine source validation pairs,
and eight separate evaluation volumes with 1,003 test slices.
Source training uses the original instance labels; evaluation uses the
specified 2D component convention.

```bash
unisem prepare-data --dataset emneuron \
  --input-root datasets/EMNeuron --output-root datasets/emneuron_prepared \
  --config configs/paper_protocol.json
```

The expected labeled source pairs are `<case>.tif` and `<case>_MaskIns.tif`.
The eight evaluation volume names are listed in the protocol.

## WaferEM

Obtain data through the [official WaferEM repository](https://huggingface.co/datasets/cyd0806/wafer_EM)
and review its access conditions. The public dataset listing declares
CC BY-NC 4.0 and gated access; source code licensing does not replace those
conditions.

```bash
unisem prepare-data --dataset waferem \
  --input-root datasets/wafer_EM --output-root datasets/waferem_prepared \
  --config configs/paper_protocol.json
```

Inputs are `<case>_inputs.h5` and `<case>_labels.h5`, with key `main`.
The frozen study uses five adaptation volumes and `wafer4_ds2` as the test
volume. The prepared arrays have shape `(125, 625, 625)`. Raw preparation
averages 2 x 2 XY blocks; label preparation uses stride-2 sampling. Do not
apply this transformation twice to arrays that are already prepared.

The current study identifies the prepared images as the **8 nm XY** protocol.
The old release scaffold's 16 nm declaration is not used here. Check the
physical spacing of any independently obtained data rather than inferring it
from an HDF5 filename. No Z-spacing-dependent claim is needed for the 2D task.

The local website bundles two small, existing WaferEM display crops for this
research preview. They are not a downloadable dataset or an independent random
sample. Their source, selection rule, and display transformations are recorded
alongside the viewer. Review asset permissions separately before public hosting.
