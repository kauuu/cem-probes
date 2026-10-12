"""
usage: run_probes.py [-h] --config config.yaml [--run_name name] [--split N]
                     [--output_dir path] [--rerun] [--rerun_probes]
                     [--force_cpu] [-p param_name value]

Inter-concept leakage probes for a trained CEM.

For the run `run_name` (trial `split`) of the given experiment config, this
script:
    1. Collects the positive, negative and mixed concept embeddings of the
       trained model on the train/val/test splits (train without augmentation).
    2. Trains a linear probe for every (target concept i, source embedding j)
       pair, as well as oracle probes fed the ground-truth label of concept j.
    3. Saves the AUC matrices M, O and the leakage M - O, and plots them with
       concepts boxed by concept group.

Everything is written to {results_dir}/probes/. The model must have been
trained beforehand with experiments/run_experiments.py using the same config.
"""
import argparse
import copy
import joblib
import json
import logging
import numpy as np
import os
import torch
import yaml

from pytorch_lightning import seed_everything

import cem.probes.embeddings as probe_embeddings
import cem.probes.linear as probe_linear
import cem.probes.plotting as probe_plotting
import cem.train.utils as utils
import experiments.experiment_utils as experiment_utils

from cem.models.construction import load_trained_model
from experiments.run_experiments import _generate_dataset_and_update_config


def _build_run_config(experiment_config, run_name, global_params=None):
    # Mirrors how experiments/run_experiments.py::main assembles a run config
    experiment_config = copy.deepcopy(experiment_config)
    shared_params = experiment_config.get('shared_params', {})
    for key, vals in experiment_config.items():
        if key not in ['runs', 'shared_params']:
            shared_params[key] = vals
    utils.extend_with_global_params(shared_params, global_params or [])

    candidates = []
    for current_config in experiment_config.get('runs', []):
        trial_config = copy.deepcopy(shared_params)
        trial_config.update(current_config)
        for run_config in experiment_utils.generate_hyperparameter_configs(
            trial_config
        ):
            run_config = copy.deepcopy(run_config)
            experiment_utils.evaluate_expressions(run_config, soft=True)
            if "run_name" not in run_config:
                run_config["run_name"] = (
                    f"{run_config['architecture']}"
                    f"{run_config.get('extra_name', '')}"
                )
            if run_config["run_name"] == run_name:
                candidates.append(run_config)
    if not candidates:
        raise ValueError(f'No run named "{run_name}" in the given config!')
    if len(candidates) > 1:
        raise ValueError(
            f'Found {len(candidates)} runs named "{run_name}" in the given '
            f'config (e.g., due to grid_variables). Please make run names '
            f'unique.'
        )
    return candidates[0]


def _concept_names(config, n_concepts):
    if config['dataset_config']['dataset'].strip().lower() == "cub":
        import cem.data.CUB200.cub_loader as cub_data_module
        names = list(
            np.array(cub_data_module.CONCEPT_SEMANTICS)[
                cub_data_module.SELECTED_CONCEPTS
            ]
        )
        if len(names) == n_concepts:
            return names
    return [f"c{i}" for i in range(n_concepts)]


def main(
    experiment_config,
    run_name,
    split=0,
    result_dir=None,
    rerun=False,
    rerun_probes=False,
    global_params=None,
    accelerator="auto",
):
    seed_everything(42)
    config = _build_run_config(experiment_config, run_name, global_params)
    result_dir = result_dir or config.get('results_dir', 'results')
    config['result_dir'] = result_dir
    config['split'] = split
    probe_config = config.get('probe_config', {})
    output_dir = os.path.join(result_dir, "probes")
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu" if accelerator == "cpu" else None

    # Train embeddings are extracted with the deterministic eval transform
    config['dataset_config']['train_augment'] = False
    (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        task_class_weights,
        _,
    ) = _generate_dataset_and_update_config(config)
    experiment_utils.evaluate_expressions(config)
    n_concepts = config['n_concepts']

    def model_fn():
        model_path = os.path.join(
            result_dir,
            f'{run_name}_fold_{split + 1}.pt',
        )
        if not os.path.exists(model_path):
            raise ValueError(
                f'Could not find trained model at {model_path}. Train it first '
                f'with experiments/run_experiments.py.'
            )
        return load_trained_model(
            config=config,
            n_tasks=config['n_tasks'],
            n_concepts=n_concepts,
            result_dir=result_dir,
            split=split,
            imbalance=imbalance,
            task_class_weights=task_class_weights,
            accelerator=accelerator,
        )

    # 1. Embeddings
    embs = probe_embeddings.load_or_collect_embeddings(
        model_fn=model_fn,
        dls={"train": train_dl, "val": val_dl, "test": test_dl},
        output_dir=output_dir,
        run_name=run_name,
        split=split,
        rerun=rerun,
        device=device,
    )

    # 2. Probes
    inputs = probe_config.get('inputs', ["pos_neg", "mixed"])
    probes_path = os.path.join(
        output_dir,
        f'{run_name}_split_{split}_probes.joblib',
    )
    if os.path.exists(probes_path) and not (rerun or rerun_probes):
        print(f"Loading cached probe results from {probes_path}")
        results = joblib.load(probes_path)
    else:
        results = probe_linear.leakage_matrices(
            embs,
            inputs=inputs,
            weight_decays=probe_config.get(
                'weight_decays',
                [0.0, 1e-4, 1e-3, 1e-2],
            ),
            epochs=probe_config.get('epochs', 300),
            learning_rate=probe_config.get('learning_rate', 0.01),
            batch_size=probe_config.get('batch_size', 4096),
            device=device,
        )
        results["concept_names"] = _concept_names(config, n_concepts)
        results["concept_group_map"] = dict(concept_map or {})
        joblib.dump(results, probes_path)

    # 3. Figures and summary
    summary = {}
    for name in inputs:
        probe_plotting.plot_leakage_figure(
            results,
            name,
            concept_map,
            path=os.path.join(
                output_dir,
                f'{run_name}_split_{split}_leakage_{name}.png',
            ),
        )
        probe_plotting.plot_group_summary(
            results,
            name,
            concept_map,
            path=os.path.join(
                output_dir,
                f'{run_name}_split_{split}_group_leakage_{name}.png',
            ),
        )
        diag = np.diag(results[name]["test_auc"])
        split_vals = probe_plotting.within_between_summary(
            results[name]["leakage"],
            concept_map,
        )
        summary[name] = {
            "mean_own_concept_auc": float(np.nanmean(diag)),
            "mean_within_group_leakage": float(np.mean(split_vals["within"]))
                if len(split_vals["within"]) else None,
            "mean_between_group_leakage": float(np.mean(split_vals["between"]))
                if len(split_vals["between"]) else None,
        }
    with open(
        os.path.join(output_dir, f'{run_name}_split_{split}_summary.json'),
        "w",
    ) as f:
        json.dump(summary, f, indent=4)
    print(json.dumps(summary, indent=4))
    print(f"Probe outputs written to {output_dir}")
    return results


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Inter-concept leakage probes for a trained CEM.',
    )
    parser.add_argument(
        '--config',
        '-c',
        required=True,
        help="YAML config the model was trained with.",
        metavar="config.yaml",
    )
    parser.add_argument(
        '--run_name',
        default="CEM",
        help="Name of the run to probe. Defaults to CEM.",
    )
    parser.add_argument(
        '--split',
        type=int,
        default=0,
        help="Trial/split of the run to probe (0-indexed). Defaults to 0.",
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        default=None,
        help="Results directory of the run. Defaults to the config's.",
    )
    parser.add_argument(
        '--rerun',
        '-r',
        action="store_true",
        default=False,
        help="Recollects embeddings and retrains probes.",
    )
    parser.add_argument(
        '--rerun_probes',
        action="store_true",
        default=False,
        help="Retrains probes but reuses cached embeddings.",
    )
    parser.add_argument(
        '--force_cpu',
        action="store_true",
        default=False,
        help="Forces CPU usage.",
    )
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        nargs=2,
        metavar=('param_name', 'value'),
        help='Overwrites a config param (same as in run_experiments.py).',
        default=[],
    )
    return parser


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    with open(args.config, "r") as f:
        loaded_config = yaml.load(f, Loader=yaml.FullLoader)
    main(
        experiment_config=loaded_config,
        run_name=args.run_name,
        split=args.split,
        result_dir=args.output_dir,
        rerun=args.rerun,
        rerun_probes=args.rerun_probes,
        global_params=args.param,
        accelerator=(
            "gpu" if (not args.force_cpu) and torch.cuda.is_available()
            else "cpu"
        ),
    )
