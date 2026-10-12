"""
Linear inter-concept probes.

For a set of k concept representations (one per concept embedding space j) we
train, for every pair (i, j), a logistic-regression probe predicting the
ground-truth concept i from representation j. All k * k probes are trained in
parallel as a single (k x d x k) weight tensor: probe (i, j) only sees
representation j and only receives the loss of concept i, so this is exactly
equivalent to training them separately.

Matrices are returned with rows indexing the TARGET concept i and columns the
SOURCE representation j, i.e., entry (i, j) measures how much concept i leaks
into embedding j. Note that this is the transpose of the convention used by
`cem.metrics.oracle.concept_purity_matrix`.
"""
import numpy as np
import scipy.stats
import torch


def batched_auc(scores, labels, chunk_size=16):
    """
    Tie-aware ROC-AUC of every probe at once.

    :param np.ndarray scores: Probe scores with shape N x k_src x k_tgt.
    :param np.ndarray labels: Binary labels with shape N x k_tgt.
    :param int chunk_size: Number of sources ranked at once (bounds memory).

    :returns np.ndarray: A k_src x k_tgt matrix of AUCs (NaN for targets with
        a single class in `labels`).
    """
    n, k_src, k_tgt = scores.shape
    labels = (np.asarray(labels) > 0.5).astype(np.float64)
    n_pos = labels.sum(axis=0)
    n_neg = n - n_pos
    auc = np.zeros((k_src, k_tgt))
    for start in range(0, k_src, chunk_size):
        chunk = scores[:, start:start + chunk_size, :]
        ranks = scipy.stats.rankdata(
            chunk.reshape(n, -1),
            axis=0,
        ).reshape(chunk.shape)
        rank_sum = np.einsum('nst,nt->st', ranks, labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            auc[start:start + chunk_size] = (
                (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            )
    auc[:, (n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def _fit(
    x_train,
    c_train,
    weight_decay,
    epochs,
    learning_rate,
    batch_size,
    device,
    seed,
):
    torch.manual_seed(seed)
    n, k_src, d = x_train.shape
    k_tgt = c_train.shape[-1]
    weights = torch.zeros(k_src, d, k_tgt, device=device, requires_grad=True)
    bias = torch.zeros(k_src, k_tgt, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idxs = perm[start:start + batch_size]
            logits = torch.einsum('nsd,sdt->nst', x_train[idxs], weights) + bias
            targets = c_train[idxs].unsqueeze(1).expand_as(logits)
            # Mean over samples and targets, sum over the independent sources
            loss = loss_fn(logits, targets) * k_src
            loss = loss + weight_decay * (weights ** 2).sum() / k_tgt
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return weights.detach(), bias.detach()


def _predict(x, weights, bias, batch_size):
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            outputs.append((
                torch.einsum('nsd,sdt->nst', x[start:start + batch_size], weights)
                + bias
            ).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def probe_auc_matrix(
    x_train,
    c_train,
    x_val,
    c_val,
    x_test,
    c_test,
    weight_decays=(0.0, 1e-4, 1e-3, 1e-2),
    epochs=300,
    learning_rate=0.01,
    batch_size=4096,
    device=None,
    seed=42,
):
    """
    Trains a linear probe for every (target concept i, source representation
    j) pair and reports its test AUC. The L2 penalty of each probe is selected
    independently using its validation AUC.

    :param np.ndarray x_train: Training representations, N x k_src x d (one
        d-dimensional representation per source concept).
    :param np.ndarray c_train: Training concept labels, N x k_tgt.
    :param np.ndarray x_val: Validation representations.
    :param np.ndarray c_val: Validation concept labels.
    :param np.ndarray x_test: Test representations.
    :param np.ndarray c_test: Test concept labels.
    :param List[float] weight_decays: L2 penalties to select from.
    :param int epochs: Number of training epochs for every probe.
    :param float learning_rate: Adam learning rate.
    :param int batch_size: Batch size.
    :param str device: Device to train in. Defaults to CUDA if available.
    :param int seed: Random seed.

    :returns Dict[str, np.ndarray]: A dictionary with
        - "test_auc": k_tgt x k_src matrix of test AUCs.
        - "val_auc": k_tgt x k_src matrix of the selected validation AUCs.
        - "weight_decay": k_tgt x k_src matrix of the selected L2 penalties.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, x_val, x_test = [
        np.asarray(x, dtype=np.float32) for x in (x_train, x_val, x_test)
    ]
    # Standardise every feature using training statistics
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1

    def to_torch(x):
        return torch.from_numpy((x - mean) / std).to(device)

    x_train_t, x_val_t, x_test_t = map(to_torch, (x_train, x_val, x_test))
    c_train_t = torch.from_numpy(
        (np.asarray(c_train) > 0.5).astype(np.float32)
    ).to(device)

    best_val = None
    for weight_decay in weight_decays:
        weights, bias = _fit(
            x_train_t,
            c_train_t,
            weight_decay=weight_decay,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        val_auc = batched_auc(_predict(x_val_t, weights, bias, batch_size), c_val)
        test_auc = batched_auc(
            _predict(x_test_t, weights, bias, batch_size),
            c_test,
        )
        if best_val is None:
            best_val = val_auc
            best_test = test_auc
            best_wd = np.full(val_auc.shape, weight_decay)
        else:
            improved = np.nan_to_num(val_auc, nan=-1) > np.nan_to_num(
                best_val,
                nan=-1,
            )
            best_val = np.where(improved, val_auc, best_val)
            best_test = np.where(improved, test_auc, best_test)
            best_wd = np.where(improved, weight_decay, best_wd)

    # Transpose to (target concept i, source representation j)
    return {
        "test_auc": best_test.T,
        "val_auc": best_val.T,
        "weight_decay": best_wd.T,
    }


def leakage_matrices(embs, inputs=("pos_neg", "mixed"), **probe_kwargs):
    """
    Computes the probe AUC matrix M for every requested embedding space, the
    oracle matrix O (same probes fed the ground-truth label of concept j) and
    the leakage matrices M - O.

    :param Dict[str, Dict[str, np.ndarray]] embs: Output of
        `cem.probes.embeddings.load_or_collect_embeddings` with "train",
        "val" and "test" entries.
    :param List[str] inputs: Embedding spaces to probe. Valid values are
        "pos_neg" (concatenation of positive and negative embeddings, 2m),
        "pos", "neg" and "mixed" (the bottleneck, m).
    :param probe_kwargs: Extra arguments for `probe_auc_matrix`.

    :returns Dict[str, Dict[str, np.ndarray]]: For every input name, the
        output of `probe_auc_matrix` plus a "leakage" entry (test AUC minus
        oracle test AUC). The oracle itself is stored under "oracle".
    """
    def _representation(split_embs, name):
        if name == "pos_neg":
            return np.concatenate(
                [split_embs["pos"], split_embs["neg"]],
                axis=-1,
            )
        if name == "oracle":
            return split_embs["c_true"][:, :, None]
        return split_embs[name]

    results = {}
    for name in ["oracle"] + list(inputs):
        print(f"Training {name} probes...")
        results[name] = probe_auc_matrix(
            x_train=_representation(embs["train"], name),
            c_train=embs["train"]["c_true"],
            x_val=_representation(embs["val"], name),
            c_val=embs["val"]["c_true"],
            x_test=_representation(embs["test"], name),
            c_test=embs["test"]["c_true"],
            **probe_kwargs,
        )
    for name in inputs:
        results[name]["leakage"] = (
            results[name]["test_auc"] - results["oracle"]["test_auc"]
        )
    return results
