# Evaluation

## Slice metrics

Prediction and GT are both converted to 2D component-pseudo instances:
each original positive ID is separated into its eight-connected components,
with minimum area one. This transformation is performed on each complete
slice before metric computation, not on a displayed crop. The official ELF
VOI and adapted Rand functions define the reported metrics.

Report VOI split, VOI merge, their sum, and ARAND. For slice `z`:

```text
VOI_sum[z] = VOI_split[z] + VOI_merge[z]
CREMI[z] = sqrt(VOI_sum[z] * ARAND[z])
```

`mean_per_slice_cremi` averages these slice scores.
`cremi_from_aggregate_components` computes the square root after aggregation.
They are different statistics and have different field names.

```bash
unisem evaluate --prediction outputs/example.h5 \
  --ground-truth datasets/example_labels.tif --output outputs/metrics.json
```

Choose `--prediction-key` if the file contains a different segmentation key.
Metrics are not computed from RGB viewer images.

## Volume and seed aggregation

For EMNeuron, average slices within each volume, then average the eight volume
means with equal weight. Do not pool all 1,003 slices into one weighted mean.
For WaferEM, average the 125 slices of the fixed test volume. For repeated
training, complete this aggregation separately for each seed, then report
the seed mean and sample standard deviation (`ddof=1`).

```bash
unisem evaluate --manifest examples/evaluation_manifest.json \
  --output outputs/emneuron_metrics.json
```

The example manifest shows the required eight volume entries. Replace the
example paths with your own existing predictions and GT. Missing volumes,
nonfinite metrics, and shape mismatches must not be silently omitted.

## Interpreting the project page

The external comparison shows means without uncertainty columns, matching the
manuscript table. The structure, stage, and efficiency comparisons retain the
available sample standard deviations. The decoder-plus-HQ update comparison
has two matched seeds; most current target routes have three. A single
checkpoint does not receive a fabricated standard deviation.

The viewer's attached numbers are existing **whole-slice** scores. The displayed
256 x 256 ROIs were selected to reveal differences and are not representative
frequency estimates. Reference-configuration qualitative comparisons use their original
model identities and do not establish the current adapter's numerical results.
