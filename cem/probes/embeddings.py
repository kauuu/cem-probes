"""
Extraction of the per-concept embeddings of a trained CEM so that they can be
probed for inter-concept leakage.
"""
import numpy as np
import os
import torch


def collect_embeddings(model, dl, device=None):
    """
    Runs a trained CEM over a dataloader (without interventions) and returns
    its concept embeddings together with the ground-truth labels.

    :param ConceptEmbeddingModel model: A trained CEM.
    :param torch.utils.data.DataLoader dl: Dataloader whose batches follow the
        usual (x, y, c, ...) format.
    :param str device: Device to run the model in. Defaults to CUDA if
        available.

    :returns Dict[str, np.ndarray]: A dictionary with the following entries
        (N samples, k concepts, m embedding size):
            - "pos": positive embeddings, N x k x m.
            - "neg": negative embeddings, N x k x m.
            - "mixed": the bottleneck fed to the label predictor, i.e.,
                p_j * pos_j + (1 - p_j) * neg_j, N x k x m.
            - "c_pred": the predicted concept probabilities p_j, N x k.
            - "c_true": the ground-truth concept labels, N x k.
            - "y_true": the ground-truth task labels, N.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    # So that pos/neg embeddings are always the first two tail outputs
    model.output_interventions = False
    model.output_latent = False
    intervention_policy = model.intervention_policy
    model.intervention_policy = None

    results = dict(pos=[], neg=[], mixed=[], c_pred=[], c_true=[], y_true=[])
    with torch.no_grad():
        for batch in dl:
            x, y, (c, _, _, _) = model._unpack_batch(batch)
            outputs = model._forward(
                x.to(device),
                train=False,
                output_embeddings=True,
            )
            c_sem, bottleneck, _, pos_embs, neg_embs = outputs[:5]
            results["pos"].append(pos_embs.detach().cpu().numpy())
            results["neg"].append(neg_embs.detach().cpu().numpy())
            results["mixed"].append(
                bottleneck.detach().cpu().numpy().reshape(pos_embs.shape)
            )
            results["c_pred"].append(c_sem.detach().cpu().numpy())
            results["c_true"].append(c.detach().cpu().numpy())
            results["y_true"].append(y.detach().cpu().numpy())

    model.intervention_policy = intervention_policy
    return {
        key: np.concatenate(vals, axis=0)
        for key, vals in results.items()
    }


def load_or_collect_embeddings(
    model_fn,
    dls,
    output_dir,
    run_name,
    split=0,
    rerun=False,
    device=None,
):
    """
    Collects (or loads from {output_dir}/{run_name}_split_{split}_embs_{name}.npz
    if already present) the embeddings of a trained CEM for every given
    dataloader.

    :param Fun[(), ConceptEmbeddingModel] model_fn: Function returning the
        trained model. Only called if something needs to be collected.
    :param Dict[str, DataLoader] dls: Dataloaders to use, keyed by split name
        (e.g., {"train": train_dl, "val": val_dl, "test": test_dl}).
    :param str output_dir: Directory where the embeddings are cached.
    :param str run_name: Name of the run the model comes from.
    :param int split: Trial/split of the run the model comes from.
    :param bool rerun: If True, ignores any cached embeddings.
    :param str device: Device to run the model in.

    :returns Dict[str, Dict[str, np.ndarray]]: The output of
        `collect_embeddings` for each dataloader in `dls`.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = None
    all_embs = {}
    for dl_name, dl in dls.items():
        path = os.path.join(
            output_dir,
            f"{run_name}_split_{split}_embs_{dl_name}.npz",
        )
        if os.path.exists(path) and (not rerun):
            print(f"Loading cached {dl_name} embeddings from {path}")
            with np.load(path) as data:
                all_embs[dl_name] = dict(data)
            continue
        if model is None:
            model = model_fn()
        print(f"Collecting {dl_name} embeddings...")
        all_embs[dl_name] = collect_embeddings(model, dl, device=device)
        np.savez(path, **all_embs[dl_name])
    return all_embs
