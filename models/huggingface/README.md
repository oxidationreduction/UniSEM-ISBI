---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - electron-microscopy
  - neuron-segmentation
  - instance-segmentation
  - unisem
  - segment-anything
datasets:
  - yanchaoz/EMNeuron
language:
  - en
---

# UniSEM — first-stage EM segmentation model

A prompt-free electron-microscopy neuron segmentation model, trained on EMNeuron
to provide reusable EM features before target specialization.

[Project page](https://oxidationreduction.github.io/UniSEM-ISBI/) ·
[Code](https://github.com/oxidationreduction/UniSEM-ISBI) ·
[Open in Colab](https://colab.research.google.com/github/oxidationreduction/UniSEM-ISBI/blob/main/notebooks/UniSEM_Colab.ipynb)

## Included assets

| File | Bytes | Role |
|---|---:|---|
| `unisem_stage1_emneuron_vitl.pt` | 1,279,754,552 | First-stage neural model |
| `unisem_stage1_emneuron_full55_rf.joblib` | 76,295,596 | Matching source-domain graph-merging RF |

The exact SHA256 values are recorded in `checksums.json`. Download both files;
the RF and its ordered 55-feature interface belong to this checkpoint's pipeline.
The RF is the full55 EMNeuron classifier paired with the source-domain result
in the current manuscript. No target-specialized WaferEM weights or training
images are included in this release.

## Architecture and provenance

- Image encoder: SAM ViT-L, initialized from the Apache-2.0 SAM model.
- Dense prediction: a prompt-free UNETR-style decoder with an HQ-inspired
  cross-layer logit-residual refinement branch. The branch does not contain
  official HQ-SAM refinement weights.
- Output channels: foreground probability, center distance, boundary distance.
- First-stage dataset: [EMNeuron](https://huggingface.co/datasets/yanchaoz/EMNeuron).
- Source checkpoint: epoch 32 from a planned 35-epoch run.
- Export: encoder and dense heads, with unused interactive SAM heads, optimizer
  state, and training paths removed. Neural states reload with `weights_only=True`.

UniSEM uses the first-stage model as the reusable starting point for a second
stage of target specialization. This download provides the first-stage model;
the paper's LightFT and FullFT target models are separate configurations.

## Run on your data

The Colab notebook installs a pinned environment, downloads the fixed model
revision, verifies both SHA256 values, and runs the public implementation.
Upload a grayscale TIFF, a ZYX TIFF stack, or an HDF5 array. It provides a real
prediction preview and exports instance labels and run metadata.

For the reference Linux environment, follow the installation instructions in
the code repository, place the two files in `artifacts/`, and run:

```bash
unisem infer --dataset emneuron \
  --config configs/paper_protocol.json \
  --checkpoint artifacts/unisem_stage1_emneuron_vitl.pt \
  --rag-checkpoint artifacts/unisem_stage1_emneuron_full55_rf.joblib \
  --input image.tif --output segmentation.h5 --device cuda
```

The workflow uses per-slice normalization, tiled multiscale prediction, watershed
fragments, and learned graph merging. The final HDF5 dataset is
`final_merged_segmentation`.

## Interpretation and limits

Predictions describe **2D instances within each slice**. Labels in adjacent
slices do not establish 3D identity. The model can retain split or merge errors,
and new acquisition domains can require target specialization. The source RF
was fitted on EMNeuron; using this demonstration on a new dataset is not a
measurement of adapted performance on that dataset.

The paper's target tables summarize multiple seeds. This source checkpoint
and the demonstration do not reproduce those target averages. The website's
WaferEM prediction explorer uses separately identified target checkpoints.

## License and attribution

The released model and source-domain RF are distributed under Apache-2.0.
Retain `LICENSE` and `NOTICE`. The SAM model is licensed under Apache-2.0;
EMNeuron's official dataset card declares MIT. The dataset is not redistributed.

This is a modified SAM-derived model trained for prompt-free EM neuron
segmentation. See `NOTICE` for upstream projects and source statements.

**RF loading:** joblib is pickle-based. Only load the paired RF after verifying
its published checksum and immutable repository revision. Do not substitute
an arbitrary uploaded joblib file.

Final publication metadata will be added when available. Please cite the
corresponding upstream works listed in the code repository.
