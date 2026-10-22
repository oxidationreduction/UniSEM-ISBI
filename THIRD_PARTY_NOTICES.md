# Third-Party Notices

UniSEM is distributed under Apache License 2.0. The following external works
retain their own copyright and license terms. This notice is informational and
does not replace the original license files or dataset agreements.

## Segment Anything

- Project: Segment Anything
- Source: https://github.com/facebookresearch/segment-anything
- ViT-L checkpoint: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- License: Apache License 2.0
- Use in UniSEM: image-encoder architecture and initialization weights

When redistributing a UniSEM checkpoint that contains SAM-derived weights,
retain the Apache-2.0 license and attribution and identify the checkpoint as a
modified derivative trained for prompt-free EM neuron segmentation.

## micro-SAM

- Project: Segment Anything for Microscopy
- Source: https://github.com/computational-cell-analytics/micro-sam
- License: MIT License
- Use in UniSEM: SAM model loading, UNETR integration, and training datasets

If micro-SAM source is copied into a release rather than installed as a
dependency, include its complete MIT license and copyright notice.

## torch-em

- Project: torch-em
- Source: https://github.com/constantinpape/torch-em
- License: MIT License
- Use in UniSEM: per-object distance targets and Dice-based distance loss

## HQ-SAM and LoRA

- HQ-SAM: https://github.com/SysCV/sam-hq (Apache-2.0)
- LoRA: https://github.com/microsoft/LoRA (MIT)
- Use in UniSEM: HQ-inspired early/late dense residual refinement and low-rank
  Q/V encoder updates. The branch is a prompt-free dense adaptation of the
  refinement idea, not a distribution of official HQ-SAM refinement weights.
  Relevant research must be cited separately from code-license notices.

## elf

- Project: elf
- Source: https://github.com/constantinpape/elf
- License: MIT License
- Use in UniSEM: segmentation and parallel-processing utilities

## Scientific Python Dependencies

UniSEM also depends on PyTorch, torchvision, NumPy, SciPy, scikit-image,
scikit-learn, h5py, ImageIO, tifffile, joblib, zarr, pooch, xxhash, vigra,
nifty, and affogato. These packages retain their respective upstream licenses. The pinned
versions are listed in `constraints.txt`.

## EMNeuron Dataset

- Dataset: EMNeuron
- Official source: https://huggingface.co/datasets/yanchaoz/EMNeuron
- Dataset-card license declaration: MIT
- Redistribution in this repository: none

Users download EMNeuron directly from the official source and are responsible
for following the dataset card and citing the corresponding publication.
The first-stage weight release reviewed the official dataset card at revision
`62df92b933550cc2a5d3849fe8543191c356f2d4`, which declares MIT and is publicly
accessible without gating. Its trained source model and source-domain RF are
distributed separately under Apache-2.0 with SAM attribution and a modification
notice; see `models/huggingface/` for the release card, checksums, and notices.

## WaferEM/MEC Dataset

- Dataset: WaferEM/MEC
- Official source: https://huggingface.co/datasets/cyd0806/wafer_EM
- Public listing: CC BY-NC 4.0, with gated access conditions
- Full dataset redistribution in this repository: none

Neither full WaferEM volumes nor prepared full-volume arrays are included.
The project page contains two existing display crops and a reference-configuration
figure for author review; they retain separate dataset/figure terms.
The source-code license does not apply to those images. Candidate target
weights are held locally until derivative-weight redistribution conditions
are resolved. No public download or permission to redistribute is claimed.

## Website

- Academic Project Page Template:
  https://github.com/eliahuhorwitz/Academic-project-page-template
- Nerfies: https://nerfies.github.io/
- Website template license: CC BY-SA 4.0
- Use: adapted academic-page organization and comparison interaction patterns;
  see the site footer and site/LICENSE.md. Template analytics and external
  runtime dependencies are not included. The website license does not cover
  third-party microscopy imagery, paper figures, datasets, or trained weights.

## WaferEM display attribution

WaferEM samples: cyd0806/wafer_EM, distributed via Hugging Face:
https://huggingface.co/datasets/cyd0806/wafer_EM

The public dataset listing identifies CC BY-NC 4.0:
https://creativecommons.org/licenses/by-nc/4.0/

The non-commercial research visualizations use documented XY mean-downsampling
of raw images, stride-2 label sampling, fixed crops, and pseudocolor instance
rendering. Actual prediction boundaries are preserved. The GIF compares existing
predictions. These assets retain dataset terms, separately from software and
website-code licenses. Access conditions can additionally apply; no model-weight
redistribution permission is inferred.

## Trained Random-Forest Artifacts

Random-forest edge classifiers are serialized with joblib and depend on
scikit-learn. Pickle-compatible files may execute code during loading. Load
only trusted artifacts listed in the local release manifest, verify their
published cryptographic checksums, and do not load untrusted files.
