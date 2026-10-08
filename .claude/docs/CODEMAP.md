# CODEMAP

A lookup guide for this repository: what each directory holds, what each file
is for, and what its classes and functions do. Line numbers (`L123`) point to
where each definition starts and may drift slightly as files change.

**Notation used below:** `B` = batch size, `k` = number of concepts
(`n_concepts`), `m` = concept embedding size (`emb_size`), `n_tasks` = number
of output classes.

---

## Table of contents

- [Repository layout](#repository-layout)
- [End-to-end flow](#end-to-end-flow)
- [`cem/models/`](#cemmodels) — model architectures
- [`cem/train/`](#cemtrain) — training & evaluation routines
- [`cem/interventions/`](#ceminterventions) — concept intervention policies
- [`cem/metrics/`](#cemmetrics) — evaluation metrics
- [`cem/data/`](#cemdata) — dataset loaders
- [`cem/utils/`](#cemutils) — misc helpers
- [`experiments/`](#experiments) — experiment runner & configs
- [Non-code directories](#non-code-directories)
- [Known issues / gotchas](#known-issues--gotchas)

---

## Repository layout

| Path | Contents |
|---|---|
| `cem/` | The installable Python package (`setup.py` → `cem`). Models, training, data, metrics, interventions. |
| `cem/models/` | All concept-based architectures (CBM, CEM, IntCEM, MixCEM, ProbCBM, PCBM, GlanceNet, …) and the config→model factory. |
| `cem/train/` | Training loops for each architecture family and the evaluation routines (task/concept accuracy, representation metrics). |
| `cem/interventions/` | Test-time concept intervention policies (random, uncertainty, CooP, behavioural cloning, …) and the intervention evaluation harness. |
| `cem/metrics/` | Accuracy/AUC helpers plus representation-quality metrics (OIS, NIS, CAS) and fairness metrics. |
| `cem/data/` | One loader module per dataset, each exposing a common `generate_data(...)` API. |
| `cem/utils/` | Small shared helpers (e.g. loading a whole DataLoader into memory). |
| `experiments/` | `run_experiments.py` (main CLI), result collection, evaluation glue, and YAML configs. |
| `examples/` | A walkthrough notebook training a CEM on the synthetic Dot dataset. |
| `figures/`, `media/` | Images used in the README; paper poster and slides. |
| `setup.py`, `requirements.txt` | Package install metadata and dependencies (Python 3.7–3.8, PyTorch Lightning < 2.0). |

---

## End-to-end flow

```
experiments/run_experiments.py  main()
  ├─ loads YAML config (shared_params + runs), expands hyper-param grids
  ├─ _generate_dataset_and_update_config()  → cem/data/<dataset>.generate_data()
  └─ for each trial/run: _multiprocess_run_trial()
        ├─ picks a train_fn by architecture  → cem/train/training.py or cem/train/train_*.py
        │     └─ cem/models/construction.py construct_model()  → model class
        │     └─ pl.Trainer.fit(...) ; saves  {result_dir}/{run_name}_fold_{split+1}.pt
        └─ experiments/evaluate_models.py evaluate_model()
              ├─ cem/train/evaluate.py evaluate_cbm()            (task/concept acc/AUC)
              ├─ cem/interventions/utils.py test_interventions() (intervention curves)
              └─ cem/train/evaluate.py evaluate_representation_metrics()  (OIS / NIS / CAS)
```

---

## `cem/models/`

### `cbm.py` — Concept Bottleneck Model (base class for almost everything)

**`ConceptBottleneckModel(pl.LightningModule)`** `L15` — Joint CBM (Koh et al. 2020). Defines the shared training loop all other concept models inherit. Subclasses customise behaviour by overriding the hook methods below.

| Method | What it does |
|---|---|
| `__init__` `L16` | Builds `x2c_model` (backbone → k logits, optionally `+extra_dims` for Hybrid-CBMs), `c2y_model` (MLP on concepts), losses, optimiser settings. |
| `_unpack_batch` `L247` | Splits a batch into `x, y, (c, g, competencies, prev_interventions)`. Supports `(x, y, c)` and `(x, (y, c))` formats. |
| `_standardize_indices` `L269` | Converts intervention indices (list / index array / binary mask, 1-D or 2-D) into a `B × k` boolean mask. |
| `_extra_losses` `L338` | Hook: extra loss terms (returns 0 here). |
| `_prior_int_distribution` `L351` | Hook: learned prior over which concept to intervene next (returns `None` here; used by IntCEM). |
| `_concept_intervention` `L364` | Replaces predicted concepts with ground truth for masked entries (handles logit-space CBMs via active/inactive intervention values). |
| `_forward` `L419` | Core forward: `x → latent → c_sem/c_pred → (interventions) → y`. Returns `(c_sem, c_pred, y_pred, *tail)`. |
| `_extra_tail_results` `L560` | Hook: extra outputs appended to the forward tuple. |
| `forward` `L571` | Public inference call (`train=False`). |
| `predict_step` `L594` | Lightning predict; honours `self.output_embeddings`. |
| `_run_step` `L615` | Computes loss = `concept_loss_weight·BCE(c_sem, c) + CE(y) + _extra_losses`, plus accuracy/AUC/F1 (and top-k) metrics. |
| `training_step` / `validation_step` / `test_step` `L716/L756/L770` | Lightning steps; log metrics (`val_`/`test_` prefixes). |
| `configure_optimizers` `L776` | Adam or SGD + optional `ReduceLROnPlateau` monitored on `loss`. |

### `cem.py` — Concept Embedding Models

**`ConceptEmbeddingModel(ConceptBottleneckModel)`** `L17` — CEM (Espinosa Zarlenga et al., NeurIPS 2022). Each concept gets a positive and a negative embedding (`m` dims each). They are mixed by the predicted concept probability, and the `k·m` bottleneck feeds `c2y_model`.

| Method | What it does |
|---|---|
| `__init__` `L18` | Builds `pre_concept_model` (backbone), one `concept_context_generators[i]` per concept (Linear → `2m` + activation), a shared or per-concept `concept_prob_generators` (Linear `2m → 1`), and `c2y_model`. |
| `_after_interventions` `L234` | Applies RandInt during training (with probability `training_intervention_prob`) and test-time interventions, then mixes embeddings into the bottleneck. Returns `(probs, intervention_idxs, bottleneck)`. |
| `_predict_labels` `L276` | Flattens the bottleneck and applies `c2y_model`. |
| `_construct_c2y_input` `L279` | `p·ĉ⁺ + (1−p)·ĉ⁻`, reshaped to `B × (k·m)`. |
| `_generate_concept_embeddings` `L298` | Backbone → per-concept contexts (`B × k × 2m`) and probabilities `c_sem` (`B × k`). Splits contexts into `pos_embs` / `neg_embs` (`B × k × m`). **This is the main entry point for extracting concept embeddings.** |
| `_new_tail_results` `L329` | Hook for subclasses to append outputs. |
| `_forward` `L340` | Full CEM forward. Returns `(c_sem, bottleneck, y_pred, *tail)`. With `output_embeddings=True` the tail ends with `pos_embs, neg_embs`. |

**`FixedEmbConceptEmbeddingModel(ConceptEmbeddingModel)`** `L457` — A CEM variant with learnable or fixed *global* per-concept embeddings (`self.concept_embeddings`, shape `k × 2 × m`). These are used either always or only for intervened concepts (`fixed_embeddings_always`). Overrides `_generate_concept_embeddings` `L573` and `_after_interventions` `L625`. Trained with `train/train_fixed_cem.py`.

### `intcbm.py` — Intervention-aware models (IntCEM, NeurIPS 2023)

**`IntAwareConceptBottleneckModel(ConceptBottleneckModel)`** `L12` — Adds a learned intervention policy (`concept_rank_model`). During training it samples intervention trajectories and penalises task errors after interventions.

| Method | What it does |
|---|---|
| `get_concept_int_distribution` `L153` | Runs a forward pass and returns the learned distribution over which concept/group to intervene on next. |
| `_construct_rank_model_input` `L192` | Input to the rank model: flattened bottleneck + previous-intervention mask. |
| `_prior_int_distribution` `L202` | Scores the next concept group with `concept_rank_model`, masking already-intervened groups (softmax at eval, raw scores in training). |
| `_concept_update_with_competencies` `L268` | Simulates imperfect experts by flipping concept labels according to `competencies`. |
| `_expected_rollout_y_logits` `L285` | Expected task logits after an intervention, averaged over expert competence outcomes. |
| `get_target_mask` `L344` | Builds the behavioural-cloning target: the concept whose intervention most increases the correct-class probability. |
| `_setup_intervention_trajectory` `L449` | Samples the initial intervention count, the trajectory horizon and the loss weights for a training step. |
| `_construct_c2y_input` `L553` | Same mixing as in CEM. |
| `_compute_task_loss` `L572` | Task loss from logits or from (probs, embeddings). |
| `_intervention_rollout_loss` `L605` | Rolls out a trajectory of interventions. Accumulates the rank-model imitation loss and the post-intervention task loss. |
| `_compute_concept_loss` `L850` | Concept BCE loss. |
| `_run_step` `L860` | Full IntCEM loss: task + concept + intervention rollout losses. |

**`IntAwareConceptEmbeddingModel(ConceptEmbeddingModel, IntAwareConceptBottleneckModel)`** `L1025` — IntCEM: CEM embeddings combined with the IntCBM training objective. Overrides `_after_interventions` `L1179` and `_prior_int_distribution` `L1240`.

### `mixcem.py` — MixCEM (ICML 2025)

**`MixCEM(IntAwareConceptEmbeddingModel)`** `L10` — Each concept's embedding = a *global* learned embedding + an input-dependent *dynamic residual*. The residual is dropped out in training (`ood_dropout_prob`) and scaled down when the concept prediction is uncertain. Interventions therefore stay effective for OOD inputs. Note: `pos_embs`/`neg_embs` here are `B × k × 2m` (global ‖ dynamic).

| Method | What it does |
|---|---|
| `__init__` `L15` | Adds global embeddings, a global context generator, Platt-scaling params, Monte-Carlo dropout settings, and the prior loss weight (`all_intervened_loss_weight`). |
| `_uncertainty_based_context_addition` `L205` | Per-concept scale for the dynamic residual based on prediction uncertainty. |
| `_predict_labels` `L215` | Predicts labels from (possibly several MC-sampled) bottlenecks and averages them. |
| `_construct_c2y_input` `L235` | Builds the mixed bottleneck(s): global + scaled dynamic residual, with random residual drop masks (MC samples). |
| `_construct_rank_model_input` `L334` | Rank-model input (dynamic + global embeddings). |
| `_new_tail_results` `L346` | Appends MixCEM-specific outputs. |
| `_generate_dynamic_concept` `L364` | Dynamic context for one concept. |
| `_generate_concept_embeddings` `L369` | Computes global + dynamic contexts and Platt-scaled probabilities, and concatenates them into pos/neg embeddings. |
| `_extra_losses` `L463` | Prior loss: task loss when *all* concepts are intervened using only the global embeddings. |
| `_concept_platt_scaling` `L507` | Per-concept affine calibration of logits. |
| `freeze_*` / `unfreeze_*` `L513–L543` | Toggle `requires_grad` for the global components, the calibration params, and the (no-op) OOD separator. Used by the stage-wise training in `train_mixcem.py`. |

### `probcbm.py` — Probabilistic CBM (Kim et al. 2023)

Adapted from the official ProbCBM repo.

| Name | What it does |
|---|---|
| `weights_init` `L27` | Weight initialiser. |
| `sample_gaussian_tensors` `L39` | Reparameterised sampling from (mu, logsigma). |
| `batchwise_cdist` `L52` | Pairwise L2 distance between multi-sample embeddings. |
| `MC_dropout` `L111` | Dropout that stays active at inference. |
| `MultiHeadSelfAttention` `L114` | Self-attention pooling module. |
| `PIENet` `L149` | Polysemous Instance Embedding module (attention + residual). |
| `MCBCELoss` `L183` | Monte-Carlo BCE loss with a KL (VIB) term over sampled concept embeddings. |
| `UncertaintyModuleImage` `L242` | Predicts the log-variance of embeddings. |
| `ConceptConvModelBase` `L274` | ResNet backbone wrapper (`forward_basic`). |
| `ProbConceptModel` `L355` | Probabilistic concept embeddings matched against learned concept vectors (`match_prob`, `get_uncertainty`, `sample_embeddings`, `forward`). |
| `ProbCBM(ProbConceptModel, ConceptBottleneckModel)` `L656` | Lightning-compatible ProbCBM. `_train_step`/`_run_step` implement the sequential concept→class training modes. `_forward` adapts the output to the CBM API. `predict_concept_dist`, `predict_concept` and `predict_class_with_gt_concepts` are inference helpers. `configure_optimizers` uses separate learning rates (`lr_ratio`). |

### `posthoc_cbm.py` — Post-hoc CBM (Yuksekgonul et al. 2022)

**`PCBM(ConceptBottleneckModel)`** `L14` — Projects a frozen black-box model's embeddings onto concept activation vectors (CAVs) and fits a sparse (elastic-net) classifier on top. An optional residual model gives PCBM-h.

| Method | What it does |
|---|---|
| `_generate_concept_scores` `L177` | Embedding (`B × m`) · CAVs (`k × m`) → concept scores. |
| `_extra_losses` `L201` | Elastic-net regulariser on the sparse classifier. |
| `freeze_non_residual_components` / `unfreeze_…` `L234/L246` | For training the residual stage. |
| `_forward` `L255` | Black-box → concept scores → classifier (+ residual). |
| `_concept_intervention` `L369` | Interventions in concept-score space. |
| `_run_step` `L415` | Loss/metrics for PCBM. |

### `glancenet.py` — GlanceNet (Marconato et al. 2022)

| Name | What it does |
|---|---|
| `update_osr_thresholds` `L14` | Computes open-set-recognition thresholds (`thr_rec`, `thr_y`) from training data. |
| `GlanceNet(ConceptBottleneckModel)` `L84` | VAE-style CBM with a decoder and a (conditional) prior. `enc_z_from_y` gives the class-conditional prior. `_extra_losses` adds reconstruction + KL. `_x2c_model` is the encoder. `_update_prediction_with_osr` applies the OSR rejection. `_forward` is the full pass. |

### `concept_to_label.py`

**`ConceptToLabelModel(pl.LightningModule)`** `L13` — Standalone concepts → label classifier (feature dropout, interventions). Used for the second stage of sequential/independent CBMs and for "C2L" runs. It has the same method set as the CBM (`_unpack_batch`, `_standardize_indices`, `_concept_intervention`, `_forward`, steps, optimisers).

### `base_wrappers.py`

| Name | What it does |
|---|---|
| `loss_from_config` `L29` | Builds a torch loss from a config dict. |
| `EvalOnlyWrapperModule(pl.LightningModule)` `L115` | Wraps any `nn.Module` for Lightning evaluation-only use (metrics, predict/val/test). |
| `WrapperModule(EvalOnlyWrapperModule)` `L283` | Adds a configurable optimiser + loss so a plain module can be trained with Lightning. |

### `standard.py` — black-box / backbone helpers

| Name | What it does |
|---|---|
| `freeze_model_weights` / `unfreeze_model_weights` `L10/L14` | Toggle `requires_grad` on all params. |
| `get_mnist_extractor_arch` `L18` | Small CNN backbone factory for MNIST-style inputs. |
| `get_out_layer_name_from_config` `L82` | Names of the output and penultimate layers per architecture (used to grab embeddings). |
| `get_latent_dim_size_from_config` `L111` | Penultimate feature size per torchvision architecture. |
| `load_vision_model` `L131` | Loads a (pretrained) torchvision model with a new output head. |
| `is_torchvision_model` `L445` | Checks whether a name refers to a supported torchvision model. |
| `construct_standard_model` `L468` | Builds a black-box model (torchvision / MLP / CNN) from a config. |

### `construction.py` — config → model factory

| Name | What it does |
|---|---|
| `LambdaLayer` `L26` | Wraps a lambda as an `nn.Module`. |
| `construct_model` `L38` | **Main factory.** Maps `config["architecture"]` (`CEM`, `IntCEM`, `MixCEM`, `CBM`, `ProbCBM`, `PCBM`, `GlanceNet`, `C2L`, `FixedEmbCEM`, `IntCBM`, …) to a class plus kwargs. Also resolves `c_extractor_arch` strings (`resnet18/34/50`, `densenet121`, `identity`, …). |
| `construct_sequential_models` `L616` | Builds separate x→c and c→y models for sequential/independent CBMs. |
| `load_trained_model` `L704` | Rebuilds a model from its config and loads `{result_dir}/{run_name}_fold_{split+1}.pt`. For logit-space CBMs/PCBMs it also estimates the intervention values from training data. **Use this to load frozen trained models (e.g. for probing).** |

---

## `cem/train/`

### `training.py` — generic training loops

| Name | What it does |
|---|---|
| `_make_callbacks` `L23` | Early stopping + best-model checkpoint callbacks from the config. |
| `_check_interruption` `L56` | Asks interactively whether to continue after a Ctrl-C. |
| `_restore_checkpoint` `L72` | Reloads the best validation checkpoint after early stopping. |
| `train_end_to_end_model` `L91` | Builds, trains (or loads cached) and saves a jointly-trained model (CBM/CEM/IntCEM/…). Saves the `.pt` file, training times, and the config `.joblib`. Returns the model plus train/val/test metrics. |
| `train_sequential_model` `L328` | Two-stage CBM: train x→c, then train c→y on the *predicted* concepts. |
| `train_independent_model` `L672` | Two-stage CBM: c→y trained on the *ground-truth* concepts. |

### Architecture-specific trainers

Each exposes one function with the same signature/return contract as `train_end_to_end_model`.

| File | Function | Notes |
|---|---|---|
| `train_blackbox.py` | `train_blackbox` `L24` | Plain DNN baseline. |
| `train_fixed_cem.py` | `train_fixed_cem` `L19` | Trains a CEM, then initialises the global embeddings from the average learned embeddings, then fine-tunes. |
| `train_glancenet.py` | `train_glancenet` `L18` | Trains, then computes the OSR thresholds. |
| `train_mixcem.py` | `train_mixcem` `L27` | Step 1: end-to-end training. Step 2: Platt-scaling calibration on the validation set. |
| `train_prob_cbm.py` | `train_prob_cbm` `L20` | Sequential concept-then-class training with weight freezing. |
| `train_pcbm.py` | `get_cavs` `L28`, `train_pcbm` `L50` | Train/load the black box → extract the embedding layer → learn CAVs (SVM) → train the sparse classifier (+ residual). |

### `evaluate.py` — evaluation routines

| Name | What it does |
|---|---|
| `evaluate_cbm` `L26` | Runs `trainer.predict` on a dataloader and returns `{dl_name}_acc_c/auc_c/acc_y/auc_y/…` (cached via `load_call`). |
| `evaluate_metric` `L146` | Extra named metrics (e.g. `mixcem_sel`). |
| `representation_avg_task_pred` `L267` | For each concept, trains a small Keras MLP to predict the *task* from that concept's embedding. Returns the mean accuracy. |
| `evaluate_representation_metrics` `L326` | Loads the trained model, extracts the bottleneck reshaped to `N × k × m`, and computes OIS (`run_ois`), the repr→task prediction (`run_repr_avg_pred`), NIS (`run_nis`) and CAS (`run_cas`). Skipped when `skip_repr_evaluation: true`. |

### `utils.py` — training utilities

| Name | What it does |
|---|---|
| `execute_and_save` `L23` | Runs a function in a spawned subprocess and caches its result to a `.joblib` file (reloads if present). |
| `load_call` `L54` | Returns cached results from `old_results` if all keys exist, otherwise calls the function. Can be forced via the env var `RERUN_METRIC_<KEY>=1`. |
| `_to_val`, `extend_with_global_params` `L91/L114` | Parse `-p key value` CLI overrides into the config. |
| `compute_bin_accuracy` / `compute_accuracy` `L125/L145` | Concept/task accuracy, AUC, F1 (binary & multiclass). |
| `wrap_pretrained_model` `L173` | Turns a torchvision constructor into a `c_extractor_arch(output_dim)` factory with a resized head. |
| `EmptyEnter` `L198` | No-op context manager (used when wandb is off). |
| `ActivationMonitorWrapper` `L209` | Wraps a trainer to periodically dump model activations on a test set during `fit`. |
| `WrapperModule(pl.LightningModule)` `L291` | Simple Lightning wrapper for training a plain module (older duplicate of `models/base_wrappers.WrapperModule`). |

---

## `cem/interventions/`

All policies implement `__call__(x, pred_c, c, y, competencies, prev_interventions, prior_distribution) → (intervention_mask, c_used)`. The models call them inside `_forward` when `self.intervention_policy` is set.

| File | Class / function | What it does |
|---|---|---|
| `intervention_policy.py` | `InterventionPolicy(ABC)` `L3` | Abstract base: stores the model, concept-group map, number of groups to intervene on, and the horizon. |
| `random.py` | `IndependentRandomMaskIntPolicy` `L5` | Randomly picks un-intervened concepts/groups (optionally weighted by the learned prior). |
| `uncertainty.py` | `UncertaintyMaximizerPolicy` `L5` | Intervenes first on the concepts with the highest predictive entropy. |
| `delta.py` | `DeltaIntPolicy` `L5` | Intervenes on a fixed, given set of concepts. |
| `global_policies.py` | `ConstantMaskPolicy` `L8` | Always uses a fixed mask. |
| | `GlobalValidationPolicy` `L42` | Global concept ordering by validation error. |
| | `GlobalValidationImprovementPolicy` `L134` | Global ordering by validation improvement when intervened. |
| `coop.py` | `CooP` `L10` | Cooperative Prediction policy: scores concepts by a weighted mix of uncertainty, task-importance and acquisition cost (`_importance_score`, `_uncertainty_score`, `_coop_step`). |
| `optimal.py` | `GreedyOptimal(CooP)` `L3` | Greedy oracle that uses ground truth to pick the best next concept. |
| `behavioural_learning.py` | `BehavioralLearningPolicy` `L17` | Learns a policy by imitating the greedy oracle (`_generate_behavioral_cloning_dataset`, `_next_intervention`). |
| `utils.py` | `InterventionPolicyWrapper` `L89`, `AllInterventionPolicy` `L120` | Adapt plain functions into policies; the latter intervenes on everything. |
| | `concepts_from_competencies` `L140` | Simulates noisy experts: flips concept labels according to competence levels. |
| | `_default_competence_generator`, `_random_uniform_competence` `L223/L236` | Competence samplers. |
| | `adversarial_intervene_in_cbm` `L248` | Intervention curve with adversarial (wrong) interventions. |
| | `intervene_in_cbm` `L317` | **Core:** loads a model and evaluates task accuracy after intervening on 0…N concept groups with a given policy. |
| | `fine_tune_coop` `L632` | Tunes CooP hyperparameters on validation data. |
| | `generate_policy_training_data` `L805` | Creates data for the behavioural-cloning policy. |
| | `get_int_policy` `L912` | Maps policy names (`random`, `uncertainty`, `coop`, `behavioural_cloning`, `optimal_greedy`, `global_val_error`, `global_val_improvement`) to configured policy classes. |
| | `_rerun_policy`, `_evaluate_intervention_auc` `L1163/L1211` | Rerun logic; area under the intervention curve. |
| | `test_interventions` `L1316` | Runs every policy in `intervention_config` for a run and returns the result dict (called from `experiments/evaluate_models.py`). |

---

## `cem/metrics/`

| File | Contents |
|---|---|
| `accs.py` | `compute_bin_accuracy` `L9`, `compute_accuracy` `L69`: concept/task accuracy, AUC and F1 (optionally per-concept AUCs). |
| `task_metrics.py` | Generic classification metrics on numpy arrays: `make_discrete`, `accuracy`, `train_weighted_accuracy`, `multilabel_accuracy`, `mean_multilabel_accuracy`, `balanced_accuracy`, `auc`, `f1`, `recall`, `precision`. |
| `test.py` | `normalize` `L13` (logits→probs), `test_metrics` `L37` (evaluate a list of metrics). |
| `fairness.py` | Group fairness metrics: `make_discrete`, `worst_group_accuracy`, `average_group_accuracy`, `worst_group_auc`, `fpr_difference`. |
| `cas.py` | `concept_alignment_score` `L11`: Concept Alignment Score — clusters each concept's representations and measures their homogeneity with the true concept labels. |
| `niching.py` | Niche Impurity Score (NIS): `niche_completeness`, `niche_completeness_ratio`, `niche_impurity`, `niche_finding` (find each concept's "niche" via MI/correlation), `niching_high_dim` (multi-dim representations), `niche_impurity_score` (main entry). |
| `oracle.py` | Oracle Impurity Score (OIS). **Closest existing code to inter-concept probing.** See below. |

**`oracle.py` functions**

| Function | What it does |
|---|---|
| `concept_similarity_matrix` `L24` | Avg normalised dot product between the representations of concept i and concept j (Concept Whitening metric). |
| `find_max_alignment` / `max_alignment_matrix` `L96/L129` | Greedy row↔column alignment of a matrix. |
| `concept_purity_matrix` `L153` | **k × k matrix: entry (i,j) = test AUC of an MLP trained to predict concept j from concept i's representation.** |
| `encoder_concept_purity_matrix` `L466` | The same, but the representations come from an encoder model applied to features. |
| `oracle_purity_matrix` `L530` | The same matrix computed from *ground-truth* concepts (the oracle baseline). |
| `normalize_impurity` `L585` | Normalises the impurity norm by the number of concepts. |
| `oracle_impurity_score` `L589` | OIS = normalised ‖purity − oracle purity‖. |
| `encoder_oracle_impurity_score` `L751` | OIS for an encoder model. |

Note: `oracle.py` and `train/evaluate.py` use **TensorFlow/Keras** for the helper MLPs.

---

## `cem/data/`

**Common loader API.** Every dataset module exposes `generate_data(config, root_dir, seed, output_dataset_vars, train/val/test_sample_transform, …)`. It returns `train_dl, val_dl, test_dl, imbalance` and, when `output_dataset_vars=True`, also `(n_concepts, n_tasks, concept_group_map)`. Batches are `(x, y, c)`. The dataset is chosen in `experiments/run_experiments.py` by `dataset_config.dataset`. Most loaders read their root from the `DATASET_DIR` env var.

| File | Dataset | Main contents |
|---|---|---|
| `CUB200/cub_loader.py` | CUB-200-2011 (112 concepts, 200 classes); also **TravelingBirds** via `traveling_birds_root_dir` | Concept/class name constants, `discrete_to_continuous_unc`, `Sampler`, `StratifiedSampler`, `CUBDataset` (pkl-based, supports uncertainty/competence and concept subsampling), `ImbalancedDatasetSampler`, `load_data`, `find_class_imbalance`, `generate_data`. |
| `CUB200/data_processing.py` | CUB preprocessing | `extract_data`: builds the train/val/test pkl metadata (from the original CBM repo). |
| `awa2_loader.py` | Animals with Attributes 2 | `AwA2Dataset`, `get_transform_awa2`, `load_data`, `get_num_labels`, `get_num_attributes`, `generate_data`. |
| `celeba_loader.py` | CelebA | `generate_data` (selected attributes as concepts/tasks). |
| `cifar_10_loader.py` | CIFAR-10 with VLM-derived concepts | `Cifar10Dataset` (`_download_data_and_splits`, `_process_vit_concepts`), `load_data`, `get_num_labels`, `get_num_attributes`, `generate_data`. |
| `waterbirds_loader.py` | Waterbirds | Attribute parsing helpers, `WaterbirdsDataset`, `get_transform_waterbirds`, `load_data`, `generate_data`. |
| `traffic_loader.py` | Synthetic traffic (PyC) | `TrafficDataset` (meta → concepts/label), `get_transform_traffic`, `load_data`, `generate_data`. |
| `mnist_add.py` | MNIST-Addition | `inject_uncertainty`, `produce_addition_set` (operand digits as concepts, sum as task), `load_mnist_addition`, `generate_data`. |
| `color_mnist_add.py` | Colour MNIST-Addition (+ SVHN domain shift) | `mixed_dataset`, `_color_digit`, `produce_addition_set`, `load_color_mnist_addition`, `generate_data`. |
| `synthetic_loaders.py` | XOR / Trig / Dot toy datasets | `generate_xor_data`, `generate_trig_data`, `generate_dot_data`, `SyntheticGenerator` (wraps them in the `generate_data` API), `get_synthetic_num_features`, `get_synthetic_data_loader`. |
| `utils.py` | Shared transforms | `gauss_noise_tensor`, `salt_and_pepper_noise_tensor`, `harder_salt_and_pepper_noise_tensor` (OOD corruptions), `LambdaDataset` (applies a transform to `x` only), `transform_from_config` (dict → transform: noise, s&p, blur, affine, randaugment, normalize, …). |

**Raw data shipped in the repo:** `cem/data/CUB200/class_attr_data_10/{train,val,test}.pkl` are the preprocessed CUB metadata splits (image paths, labels, attributes). `selected_*_sampling_*.npy` files are fixed concept/group subsets for the "incomplete" experiments. Image data itself must be downloaded separately.

---

## `cem/utils/`

| File | Contents |
|---|---|
| `data.py` | `_largest_divisor` (batch-size helper), `daloader_to_memory` `L13`: loads an entire DataLoader into `(x, y, c[, g])` arrays/tensors (`only_labels=True` skips `x`). |

---

## `experiments/`

| File | Contents |
|---|---|
| `run_experiments.py` | **Main CLI** (`python experiments/run_experiments.py -c <config.yaml>`). |
| `evaluate_models.py` | `evaluate_model` `L9`: per-run evaluation glue. Runs `evaluate_cbm` on every test set (incl. OOD `additional_test_sets`), the additional metrics, `test_interventions`, and `evaluate_representation_metrics`. |
| `experiment_utils.py` | Config and result helpers (see below). |
| `collect_results.py` | CLI to summarise finished experiments, do model selection, and emit a new config with only the selected runs (`construct_selected_config` `L174`, YAML dump helpers `FlowList`, `IndentDumper`, `QuotedString`, …). |

**`run_experiments.py` functions**

| Function | What it does |
|---|---|
| `update_statistics` `L144` | Merges results into an aggregate dict with a prefix. |
| `hash_function` `L153` | Hashes a function's bytecode (for dataset-cache keys). |
| `_apply_transformation` `L159` | Wraps a DataLoader with a transform (used for OOD test sets). |
| `_update_config_with_dataset` `L170` | Writes `n_concepts`, `n_tasks`, `concept_map` and `input_shape` into the config; computes task class weights. |
| `_generate_dataset_and_update_config` `L229` | Dispatches on `dataset_config.dataset` to a loader module, applies transforms, builds the backbone `c_extractor_arch`, and caches datasets. |
| `_perform_model_selection` `L553` | Picks the best hyperparameter config per group on validation metrics after `model_selection_trials` trials. |
| `_multiprocess_run_trial` `L622` | Picks the training function by architecture, trains, then evaluates (in a subprocess). |
| `main` `L853` | Loops trials × runs × hyperparameter grid, applies filters/rerun logic, saves `results.joblib`, prints summary tables. |
| `_build_arg_parser` `L1232` | CLI arguments (`-c`, `-o`, `-p key value`, `--filter_in/out`, `--rerun`, …). |

**`experiment_utils.py` functions**

| Function | What it does |
|---|---|
| `determine_rerun` `L17` | Decides whether a run must be retrained (from config/env flags). |
| `get_mnist_extractor_arch` `L37` | CNN backbone for MNIST-style data. |
| `get_metric_from_dict` `L96` | Pulls a metric (mean/std) out of nested results. |
| `perform_model_selection` / `perform_averaging` `L107/L168` | Select the best run per group / average across groups. |
| `print_table` `L210` | Pretty-prints result tables (mean ± std over trials). |
| `filter_results` `L380` | Keeps results whose keys mention a run name. |
| `evaluate_expressions` `L390` | Resolves `"{other_key}"` templates and `"{{ expr }}"` Python expressions inside config values. |
| `initialize_result_directory` `L415` | Creates `models/` and `history/` subdirectories. |
| `has_/get_hierarchical_key`, `flatten_dictionary`, `nested_dictionary_set` `L431–L469` | Dotted-key access helpers for nested configs. |
| `generate_hyperparameter_configs` `L482` | Expands list-valued config entries into the Cartesian product of run configs. |

### `experiments/configs/`

YAML experiment definitions. Each has `shared_params` (`results_dir`, `trials`, `dataset_config`, `intervention_config`, `eval_config`, training params) and a `runs:` list (one entry per model, with `architecture`, `run_name` and overrides; list values = hyperparameter grid). Placeholders such as `root_dir: "/path/to/..."` must be filled in.

| Path | Contents |
|---|---|
| `configs/*.yaml` | Main experiments per dataset (`cub`, `awa2`, `celeba`, `cifar10`, `mnist_add`, `travelingbirds`, synthetic `xor`/`trig`/`dot`). `*_incomplete.yaml` = concept-incomplete variants. |
| `configs/hyper_searches/` | Hyperparameter sweep versions of the main configs. |
| `configs/additional_experiments/ablations/` | Completeness and noise ablations (CUB, AwA2, MixCEM). |
| `configs/additional_experiments/mnist_domain_shift_experiments/` | Colour-MNIST → SVHN domain-shift experiments. |

---

## Non-code directories

| Path | Contents |
|---|---|
| `examples/dot_cem_train_walkthrough.ipynb` | Tutorial: load the Dot dataset, build a CEM, train it with Lightning, extract concept/embedding predictions, compute metrics. |
| `figures/` | Architecture diagrams (CEM, IntCEM, MixCEM) used in the README. |
| `media/` | Paper poster (PDF) and slides (PPTX). |
| `results/` (created at runtime) | Per-experiment outputs: `{run}_fold_{n}.pt` checkpoints, `*_training_times.npy`, `*_experiment_config.joblib`, cached metric `.joblib` files, `results.joblib`. |

---

## Known issues / gotchas

- `experiments/run_experiments.py:113` imports `cem.data.siim_arc_loader`, but that module does not exist in this repo, so the import fails at startup unless the import is removed or the module is added.
- `cem/metrics/oracle.py` and `cem/train/evaluate.py` import `tensorflow`, and `cem/metrics/cas.py` imports `sklearn_extra`. `tensorflow` is not listed in `requirements.txt`/`setup.py`. Since every trainer imports `evaluate.py`, both packages are required for any training run. `environment.yml` (Mac, osx-64) and `environment.cuda.yml` (cluster) pin compatible versions.
- In `ConceptEmbeddingModel.__init__`, `inactive_intervention_values` defaults to **ones** (not zeros). This doesn't affect CEM interventions, which use `c_true` directly.
- `cem/models/base_wrappers.py` and `cem/train/utils.py` both define a `WrapperModule`.
- MixCEM's `pos_embs`/`neg_embs` are `B × k × 2m` (global ‖ dynamic), unlike CEM's `B × k × m`.
