# Release validation

This page records implementation checks, not new scientific experiments.

## Product page and Colab preparation — 2026-10-22

- The redesigned page was inspected in a real browser at desktop, 768-pixel,
  and 390-pixel widths. A 640-pixel reflow check covers the layout width of a
  1280-pixel viewport at 200% zoom. No full-page horizontal overflow was found.
  Body text is 20px on desktop and 18px on mobile; controls/tables are at least
  16px and supporting text at least 14px.
- Actual case selection, pointer divider dragging, keyboard split adjustment,
  opacity, zoom/reset, GIF playback/pause, and keyboard evidence tabs passed.
  The page retains the shared Pointer Events implementation for touch; no
  physical touchscreen test is claimed.
- Twenty-three focused tests passed: upload/export and invalid inputs,
  published/local model manifest separation, stored RF feature order, and
  existing RF training and merge behavior.
- One real first-stage inference completed on the fixed 256px display crop
  in the reference Linux Python 3.10 environment. It produced finite dense
  arrays, uint32 final labels, a boundary preview, and the downloadable ZIP.
  This was a software integration check on one image, not a new evaluation.
- The source RF stores the same 55 named features in a different order from
  the public feature extractor. Loading now maps columns by their names;
  the current canonical-order RF interface is preserved. On 12 real edges,
  wrapper probabilities matched the original RF after manual column ordering
  exactly (maximum absolute difference zero).
- The official micromamba binary downloaded and ran on Linux. The separate
  cold-install attempt stopped because the test host could not resolve the
  upstream package hosts. The successful inference inherited the reference
  ML dependencies; it does not establish a clean Colab installation.
- Hugging Face upload and the actual Colab cold-start test remain pending
  browser access. The notebook's model revision must be filled from the
  verified upload before this prepared update is published.

## Model package

- Built and installed `unisem==0.2.0` as a wheel in a separate Python 3.10
  virtual environment. The environment inherits the existing pinned native/ML
  dependencies; this is not a fresh network or Conda dependency-solver test.
- The independent installation exposes all documented CLI commands.
- The package CPU suite passed **36 tests**. It covers structural targets,
  metric aggregation, strict checkpoint/adapter handling, freeze policies,
  inference interfaces, and RF shape, selected-slice, and test-set checks.
- Three real exported models loaded strictly on CPU. First-stage and FullFT
  contain 319,902,662 parameters; current LightFT trains 13,197,254 parameters
  with 96 LoRA tensors in 24 encoder blocks.
- Reconstructing the adapter with its hash-bound base reproduced all original
  model tensors exactly. Exported neural payloads reload in weights-only mode.

## Real-model forward check

On a fixed 256 x 256 crop, the public LightFT model and the formal model
wrapper produced finite `(1, 3, 256, 256)` predictions on CPU, with two threads.
Maximum and mean absolute differences were both zero. This was one code
parity smoke check, not new dataset inference or scientific evidence.
Official SAM initialization was also checked on CPU. CUDA training uses
FP16 AMP in the formal runs; dense inference uses FP32.

## Frozen-cache equivalence

One existing WaferEM slice was used for a CPU post-processing equivalence
check, without new neural inference or model selection. Across all three
scales, watershed fragments matched the frozen implementation pixel for
pixel. All 799 x 55 edge features matched exactly (maximum absolute difference
zero), and the matching RF produced the same 366 final instances with zero
label differences. This is a code-equivalence check on one cache, not a new
dataset evaluation or a statement about all possible inputs.

## Viewer and local assets

- All 12 display crops match the original frozen pixel hashes.
- Four categories of evidence are extracted from the author-approved tables;
  seed counts and available standard deviations are retained.
- The comparison GIF contains 40 frames totaling 10 seconds.
- The website uses local scripts, styles, images, PDFs, and JSON only. External
  links are citations or navigation targets, not required runtime resources.
- Desktop (1912 x 994) and mobile (390 x 844) views were checked in a real
  browser. Case/method switching, pointer dragging, keyboard split adjustment,
  opacity, zoom/reset, and GIF play/pause worked. No physical touch device was
  used; touch handling was inspected through the shared Pointer Events path.
- All six local model/RF assets passed byte-size and SHA256 verification. The
  temporary transfer service was stopped after delivery.
- Reaggregating the frozen eight-volume records reproduced mean slice CREMI
  0.737661 for UniSEM and 0.787331 for SegNeuron, with 1,003 slices each.

## Rerun checks

```bash
python -m pytest -q
python tools/package_release.py
python tools/check_release.py
python tools/check_release.py --check-artifacts
```

The last command hashes large local artifacts and requires the candidate
models to be present. It does not download them or run the neural model.
The project-page build helper is optional; rebuilding author assets requires
Pillow and Poppler plus an explicit `--paper-root` pointing to your authorized
manuscript/figure sources. Browsing the packaged site requires none of these.
