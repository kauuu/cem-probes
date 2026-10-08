# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Master's thesis built on top of the Concept Embedding Models (CEM) codebase.

**Hypothesis:** test-time interventions on CEMs are less effective than they
could be because of **inter-concept leakage**. When concept `i` is intervened
on, the other `k-1` concept embeddings may still carry conflicting information
about concept `i`, which undermines the intervention. Supporting evidence:
_Towards Robust Metrics for Concept Representation Evaluation_ shows that
impure concept representations degrade intervention performance (they measure
impurity model-wide via OIS; we work per concept embedding).

We do **not** try to remove leakage during training, since it may reflect
genuine correlations between concepts. We act on it only at intervention time.
The question is how much we can get out of a single intervention.

**Method:**

1. For each concept `i`, train `k-1` probes that predict concept `i` from each
   other concept's embedding `j != i`. High probe performance means concept `i`
   leaks into embedding `j`.
2. When concept `i` is intervened on, propagate the intervention to the leaky
   embeddings `j`. Do not erase the concept-`i` direction (e.g. LEACE or
   iterative erasure until the probe fails). **Replace** it with the mean
   activation along that direction.
3. Evaluate: task accuracy vs. number of concepts intervened on should rise
   faster than baseline intervention curves, and model performance should stay
   high.

## Codebase

- Look at `.claude/docs/CODEMAP.md` for codebase documentation (layout, end-to-end
  flow, every module/class/function, known gotchas). Read it before exploring
  the code.
- When adding, renaming or removing modules, classes or public functions,
  update `.claude/docs/CODEMAP.md` in the same change.
- Match the existing style. Reuse the existing intervention harness
  (`cem/interventions/`) and metrics (`cem/metrics/`, e.g. OIS) instead of
  writing parallel versions.

## Commands

There is no test suite, linter or build step (`cem/metrics/test.py` is a
metrics module, not tests). Sanity-check changes with a small local script or
the synthetic `dot`/`xor`/`trig` configs, which need no downloaded data.

```bash
CONDA_SUBDIR=osx-64 conda env create -f environment.yml   # local Mac; cluster: environment.cuda.yml
python experiments/run_experiments.py -c experiments/configs/dot.yaml           # full experiment
python experiments/run_experiments.py -c <cfg> --filter_in "CEM" -p max_epochs 2  # one run, quick override
```

Useful `run_experiments.py` flags: `-o` (results dir), `-p key value`
(override any config key, repeatable), `--filter_in/--filter_out` (regex on
`{run_name}_split_{n}`), `--rerun`, `--force_cpu`, `--fast_run`,
`--no_new_runs`. Dataset configs contain `root_dir: "/path/to/..."`
placeholders that must be filled in, or set `DATASET_DIR`.

Imports that will fail out of the box: `experiments/run_experiments.py`
imports a missing `cem.data.siim_arc_loader`. TensorFlow (missing from
`requirements.txt`) and scikit-learn-extra are imported at module level by
`cem/train/evaluate.py`, which every trainer imports, so both are hard
requirements (both env files include them; see the pin notes there).

## Architecture (big picture)

`.claude/docs/CODEMAP.md` has the full map; these are the cross-file ideas that are
easy to miss:

- **Config-driven pipeline.** `run_experiments.main` loops trials × `runs` ×
  hyperparameter grid (list-valued keys expand to a Cartesian product).
  `_multiprocess_run_trial` builds data, picks a trainer by `architecture`
  (CEM uses `cem/train/training.py::train_end_to_end_model`), and then calls
  `experiments/evaluate_models.py::evaluate_model` (accuracy, intervention
  curves, OIS/NIS/CAS).
- **Model hierarchy.** Every concept model subclasses
  `ConceptBottleneckModel` (`cem/models/cbm.py`), which owns the Lightning
  steps and loss. Subclasses override hooks: `_generate_concept_embeddings`,
  `_after_interventions`, `_construct_c2y_input`, `_extra_losses`,
  `_prior_int_distribution`, `_new_tail_results`. New architectures register
  in `cem/models/construction.py::construct_model`.
- **Interventions run inside the forward pass.** At test time, setting
  `model.intervention_policy` makes `_forward` ask the policy for a B × k
  mask whenever `c` is given and `intervention_idxs` is `None`. In a CEM,
  `_after_interventions` swaps intervened probabilities for labels before
  mixing positive and negative embeddings; the embeddings themselves are never
  changed. `cem/interventions/utils.py::intervene_in_cbm` builds curves by
  reloading the model from disk and feeding each round's mask back in as
  `prev_interventions`.
- **Forward output tuple.** `(c_sem, bottleneck, y_pred, *tail)`. The tail
  depends on flags: `output_interventions` → mask, `output_latent` → latent,
  `output_embeddings` → `pos_embs, neg_embs`. Callers index into it by
  position, so adding tail outputs can shift indices.
- **Aggressive caching.** Trained weights
  (`{result_dir}/{run_name}_fold_{split+1}.pt`), per-run results
  (`{run_name}_split_{n}_results.joblib`) and individual metrics (via
  `load_call`) are reused when present. Pass `--rerun` or set env vars
  (`RERUN_METRIC_<KEY>=1`, `RERUN_INTERVENTIONS=1`,
  `RERUN_INTERVENTION_<POLICY>=1`, `IGNORE_INTERVENTION_<POLICY>=1`) to force
  or skip work. Code changes alone do not invalidate the cache.
- Other env vars: `MULTIPROCESS=1` (each run in a spawned subprocess),
  `INT_BATCH_SIZE`, `VERBOSE_INTERVENTIONS=1`, `LOGLEVEL`.

## Workflow rules

- **Never run git commands that modify state** (commit, push, branch, rebase,
  etc.) unless explicitly asked. **Never push.** The user handles git.
- **Never ssh into the cluster** and never submit or run cluster jobs. The
  user runs all training and experiment jobs. Claude writes and fixes code and
  configs, and may run small local sanity checks.
