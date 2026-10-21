# Train a dataset specialist

First-stage training learns a complete prompt-free EM dense predictor.
Second-stage training specializes that model on target data. These stage names
refer to neural training; training a random forest is a separate operation.

## First-stage model

```bash
unisem train --profile emneuron-stage1 \
  --data-root datasets/emneuron_prepared \
  --sam-checkpoint artifacts/sam_vit_l_0b3195.pth \
  --config configs/paper_protocol.json --output-root outputs/stage1
```

This updates the encoder, base dense decoder, and HQ-inspired residual branch.
The frozen source checkpoint used for target initialization comes from a
35-epoch training run, selected using source validation data.

## LightFT

```bash
unisem train --profile waferem-lightft \
  --data-root datasets/waferem_prepared \
  --sam-checkpoint artifacts/sam_vit_l_0b3195.pth \
  --init-checkpoint artifacts/unisem_stage1_emneuron_vitl.pt \
  --seed 12345 --label-layers 160 \
  --config configs/paper_protocol.json --output-root outputs/lightft
```

LightFT freezes original encoder weights and trains rank-16 Q/V LoRA in every
one of the 24 encoder blocks, together with the dense decoder and HQ branch.
The current update set contains 13,197,254 parameters: about 4.13% of FullFT's
319,902,662. This is an update-parameter ratio, not a measured speedup or memory
reduction. The older decoder/HQ-only strategy is named `head_only` in the code.

For the frozen WaferEM study, budgets are 160, 315, or 625 labeled slices.
Every budget constrains neural training, training monitoring, and RF fitting.
The current matched schedule uses 9,375 optimizer updates; fewer labels do not
automatically reduce the number of updates. FullFT uses gradient accumulation
to match the effective batch. CUDA training uses FP16 automatic mixed precision
with a gradient scaler; dense inference uses FP32. Training-monitor crops can overlap training
crops; this is not independent target validation.

The command writes the actual label selection so that paired initializations
can reuse it with `--label-manifest`. Check the generated selection before
running a paired comparison. Label budgets and stage controls are not claims
of statistical equivalence or monotonic improvement for every seed.

## FullFT and the graph model

Use `--profile waferem-fullft` to update all original model parameters. Both
strategies still require a graph model fitted on their permitted target labels
to reproduce the complete reported instance-generation pipeline:

```bash
unisem train-rag --manifest examples/rag_manifest.json \
  --output artifacts/my_target_rf.joblib --workers 4
```

Each record names existing multiscale dense predictions, raw data, GT, and the
permitted `z_indices`. Paths are relative to the manifest. The example selects
only two slices to illustrate the schema; replace it with the complete allowed
label selection from neural training, keeping `label_budget` and
`label_z_lists` consistent. WaferEM records require explicit selected indices;
test-set records, shape mismatches, and invalid indices are rejected.
Never include the final test volume in this manifest. The 55-feature order,
RF settings, and merge threshold are fixed in the reference configuration.
Do not substitute a differently trained RF solely because its filename looks
similar. The source model and each target route have distinct matching RFs.
