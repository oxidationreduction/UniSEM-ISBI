# UniSEM models

The **first-stage EMNeuron model and matching full55 RF** are released through
[dreamBuilder/UniSEM](https://huggingface.co/dreamBuilder/UniSEM) under Apache-2.0.
The [Colab notebook](https://colab.research.google.com/github/oxidationreduction/UniSEM-ISBI/blob/main/notebooks/UniSEM_Colab.ipynb)
downloads both files at a fixed revision and verifies their SHA256 values.
WaferEM LightFT and FullFT assets remain **local candidates**. Their dataset
and derivative-weight conditions are recorded separately; they are not included
in the public model download.

## Choose a model

| Asset | Intended use | Required companion |
|---|---|---|
| First-stage EMNeuron model | Reusable EM initialization; source-domain segmentation | EMNeuron full55 RF for the reported source pipeline |
| WaferEM LightFT adapter, seed 12345 | Target specialization with the original encoder frozen | The exact sanitized first-stage model and LightFT RF |
| WaferEM FullFT model, seed 12345 | Target specialization updating the full neural model | FullFT RF |

Downloaded files and local candidates reside in the repository's ignored
`artifacts/` directory. See [manifest.json](manifest.json) for exact filenames,
byte sizes, SHA256 values, published download identities, and paired assets.
The adapter is not a standalone model. Its `base_sha256`
binds the sanitized first-stage **file**, while `source_sha256` identifies the
original training export. These hashes have different purposes.

## Architecture and training

UniSEM uses a SAM ViT-L image encoder and a prompt-free UNETR-style dense decoder,
with an HQ-SAM-inspired residual refinement branch. The branch borrows the design
idea of combining early and late encoder features with mask features; these
models do **not** contain pretrained HQ-SAM refinement weights. Dense channels
are foreground probability, center distance, and boundary distance, in that order.
The neural exports contain the image encoder and dense heads. Unused interactive
SAM prompt/mask decoder states, optimizer state, and training paths are omitted.

First-stage training uses the EMNeuron training collection to establish reusable
EM segmentation features. The selected source export comes from epoch 32 of a
35-epoch training run. Second-stage training specializes the complete source model
to WaferEM-8nm using the five designated training volumes (625 labeled slices).
LightFT trains the dense decoder, the HQ-inspired branch, and Q/V LoRA in all
24 encoder blocks. LoRA has rank 16, alpha 16, dropout 0, and 96 tensor parameters.
The original encoder weights stay frozen. The neural update comprises
13,197,254 parameters, 4.13% of FullFT's 319,902,662 updated parameters.
This ratio measures updated parameters; it is not a measured speed or memory ratio.

Both target routes use AdamW, initial learning rate 1e-5, weight decay 1e-4,
and 9,375 optimizer updates. LightFT uses physical batch 4. FullFT uses physical
batch 2 with accumulation 2, giving effective batch 4. Model selection uses
budget-covered training-monitor crops, which may overlap training crops.
Consult the paper and reproducibility configuration for the full protocol.

## RF pairing and output semantics

Neural predictions are fused within 256-, 384-, and 512-pixel window scales.
The 384-scale watershed fragments define graph nodes. A route-specific RF uses
the ordered 55 edge features; merge probability is at least 0.65, followed by
removal of instances smaller than 16 pixels. RF fitting is separate from neural
training. Do not exchange an RF between a different checkpoint or feature schema.

The first-stage model is paired with the **July 24 full55 EMNeuron RF** used by
the current source-domain table, rather than an earlier RF. The two target
models use their respective final WaferEM RFs. RF files are byte-for-byte copies
to preserve experimental identity. **Joblib is a pickle-based format: load only
trusted files after verifying their manifest hashes.** It can execute code and
is not interchangeable with the safe tensor-only neural checkpoint format.

## Representative checkpoint versus paper averages

The local target exports use seed **12345**, fixed to match existing display
caches; it was not chosen by test score. Paper results average seeds 12345,
23456, and 34567. This single exported checkpoint does not reproduce that mean
by itself. All six original target checkpoint/RF identities are retained in
[seed-identities.json](seed-identities.json); only seed 12345 is bundled locally.
Target initialization, full-label LightFT, and repeated table endpoints can share
one checkpoint and do not constitute extra independent runs.

Evaluation is 2D and section-wise, with eight-connected component-pseudo labels
for both prediction and ground truth. EMNeuron aggregates slices within each
test volume and then weights its eight test volumes equally. Target results use
the fixed 125-slice WaferEM-8nm test block. The reported mean section-wise CREMI
computes the geometric score on each slice before averaging; it is distinct from
the geometric score of aggregate VOI and ARAND. No 3D connectivity improvement
or universal cross-domain performance is implied.

## Inspect or reconstruct locally

```python
from unisem.checkpoints import load_inference_checkpoint, reconstruct_adapter

base = "artifacts/unisem_stage1_emneuron_vitl.pt"
stage1 = load_inference_checkpoint(base)
lightft = reconstruct_adapter(
    "artifacts/unisem_lightft_waferem_seed12345.adapter.pt", base
)
fullft = load_inference_checkpoint(
    "artifacts/unisem_fullft_waferem_seed12345.pt"
)
```

The export tool verifies source hashes, removes training configuration and private
paths, writes tensor-only neural states, reloads with `weights_only=True`, and
compares every restored encoder and decoder tensor exactly. Adapter construction
also verifies that every frozen encoder tensor matches the first-stage base.
It does not run training or new image inference.

## Limits and release status

These research models predict 2D neuron instances. Crowded borders and local
merges remain possible, including in the displayed residual-error case.
Source-domain performance and target specialization are separate experimental
settings. Real deployments should validate their own acquisition domain,
resolution, foreground/background conventions, and downstream instance protocol.
Training data are not redistributed. The first-stage source review covers
EMNeuron's official MIT declaration and SAM's Apache-2.0 model license; the
public model card retains the relevant attribution and modification notices.
Final author information and target-weight redistribution review remain pending.
