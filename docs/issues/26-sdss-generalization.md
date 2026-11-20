---
id: 26
title: "[Stretch] Generalization to the low-resolution SDSS Galaxy10 dataset"
owner: eftelya
labels: [area:experiments, type:experiment, priority:stretch]
milestone: "M3 - Analysis"
week: W5
depends_on: [19, 22]
---
## Goal
Test whether our best models transfer to the original 69×69 SDSS Galaxy10. This is the "if time permits" item in the proposal. **Start only after #25 is in good shape.**

## Context
The SDSS Galaxy10 (21,785 images, 69×69, astroNN `Galaxy10.h5`) has **different classes** from DECaLS. Write the class mapping carefully and evaluate only on classes that map cleanly.

## Tasks
- [ ] Download/loader in `galaxy/data/sdss.py`. Document the class mapping table SDSS ↔ DECaLS (and the classes that have no counterpart) in the report.
- [ ] Preprocessing: upsample 69 → 256 (bicubic), then the standard pipeline. Note the band difference (SDSS g/r/i vs DECaLS g/r/z) as a domain shift.
- [ ] Zero-shot evaluation of the best deep model, escnn and best classical model on the mapped SDSS classes: accuracy + macro-F1 on the mapped subset.
- [ ] *(Optional)* A short fine-tune (5 epochs, linear probe) on a small SDSS training subset, to measure adaptation.
- [ ] Connect the result to E5 downsampling: does `downsample ×4` predict SDSS performance?
- [ ] Short report subsection.

## Acceptance criteria
- Numbers + mapping table in `report/sections/sdss.md`, or a clear note on why the result is inconclusive.
