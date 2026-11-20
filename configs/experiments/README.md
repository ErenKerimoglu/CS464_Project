# Experiment configs

One sub-folder per experiment, matching the `experiment` names in PROJECT_PLAN §4.5:
`baselines_classical/`, `model_family/`, `e1_orientation/`, `e2_transfer/`, `e3_data_eff/`, `e4_imbalance/`.

An experiment config inherits from a (tuned) model config and changes only what the experiment varies:

```yaml
# configs/experiments/e1_orientation/resnet50_none.yaml
defaults: [../model_family/resnet50.yaml]   # tuned config written by #15
experiment: e1_orientation
run_name: resnet50_none
preprocess: {policy: none}
```

Evaluation-only experiments (E5 robustness, E1b rotation sweep, E6, SDSS) need no config. They run
`scripts/evaluate.py` on existing run folders.
