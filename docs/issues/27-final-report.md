---
id: 27
title: "[Report] Final report & presentation (everyone)"
owner: all
labels: [type:report, priority:critical]
milestone: "M4 - Report"
week: W6
depends_on: [17, 18, 19, 20, 21, 22, 23, 24, 25]
---
## Goal
A complete, reproducible final report and presentation built from committed results.

## Section ownership (each owner writes and polishes their sections; Eren compiles and edits)
| Section | Owner |
|---------|-------|
| Abstract, Introduction, Related Work, Conclusion | Eren (all review) |
| Dataset, deduplication, preprocessing, geometry protocol | Eren |
| Experimental setup (training recipe, tuning, seeds, metrics, significance, compute) | Ege |
| Classical methods & results (features, ablation, classical orientation) | Yunus |
| Deep model families + transfer learning (E2) | Berkay |
| Equivariant CNN | Eftelya |
| E1 Orientation (headline) | Eren |
| E3 Data efficiency | Ege |
| E4 Class imbalance | Eftelya |
| E5 Robustness | Yunus |
| E6 Error analysis (+ SDSS if done) | Eftelya |
| Limitations & future work | Berkay |
| Presentation slides | Each owner makes the slides for their sections. Berkay + Ege assemble and rehearse with everyone. |

## Tasks
- [ ] Agree on the template (course LaTeX template or Overleaf) and page limit. Put the sources in `report/`.
- [ ] Freeze results: tag `results-final` on `main`, then regenerate all tables/figures with `python scripts/aggregate_results.py` and the figure scripts.
- [ ] Every number in the text comes from `results/tables/` (no hand-typed numbers).
- [ ] Internal review: every section is read by one other member (reviewer = next person in the team list).
- [ ] Each member's contribution statement.
- [ ] Rehearse the presentation once with timing.

## Acceptance criteria
- Report PDF + slides submitted. The repo README links them and the `results-final` tag.
- Project-level definition of done (plan §10) is fully ticked.
