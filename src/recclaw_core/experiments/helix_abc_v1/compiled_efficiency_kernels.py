"""Compiler-owned scalable kernels for mechanical BL-ICF substrates.

The mechanism program owns *what* relation or sampling policy is requested.
This module owns only the stable execution substrate: sparse candidate
enumeration, bounded scoring memory, padding/positive masking, and epoch-cache
lifecycle.  Optional callbacks retain the candidate's scientific weighting and
hardness policy without allowing it to replace those mechanical boundaries.
"""

from __future__ import annotations

from typing import Any, Callable


def build_item_item_cooccurrence_topk(
    dataset: Any,
    *,
    top_k: int,
    weight_mode: str = "cosine",
    weight_transform: Callable[[Any, Any, Any], Any] | None = None,
) -> Any:
    """Build a train-only sparse item relation without an item-by-item loop.

    ``weight_transform`` receives the non-zero co-occurrence counts and the
    corresponding left/right item frequencies.  It may change edge weights,
    but sparse enumeration and row-top-k truncation remain compiler-owned.
    """

    import numpy as np

    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("item-item co-occurrence top_k must be positive")

    interactions = dataset.inter_matrix(form="csr").tocsr().astype(
        np.float32,
        copy=True,
    )
    interactions.sum_duplicates()
    # The primitive is an interaction-incidence relation, not a rating-count
    # product.  Repeated observations must not square their multiplicity.
    interactions.data.fill(1.0)
    interactions.eliminate_zeros()
    frequencies = np.asarray(interactions.sum(axis=0)).reshape(-1).astype(
        np.float32,
        copy=False,
    )

    cooccurrence = (interactions.transpose().tocsr() @ interactions).tocsr()
    cooccurrence.setdiag(0.0)
    cooccurrence.eliminate_zeros()

    output_rows: list[int] = []
    output_columns: list[int] = []
    output_values: list[float] = []
    for row_id in range(int(cooccurrence.shape[0])):
        start = int(cooccurrence.indptr[row_id])
        end = int(cooccurrence.indptr[row_id + 1])
        columns = cooccurrence.indices[start:end]
        counts = cooccurrence.data[start:end].astype(np.float32, copy=False)
        # RecBole reserves item id zero for padding.
        keep = columns != 0
        columns = columns[keep]
        counts = counts[keep]
        if row_id == 0 or columns.size == 0:
            continue

        left = np.full(counts.shape, frequencies[row_id], dtype=np.float32)
        right = frequencies[columns]
        if weight_transform is not None:
            weights = np.asarray(
                weight_transform(counts.copy(), left, right),
                dtype=np.float32,
            ).reshape(counts.shape)
        elif weight_mode == "count":
            weights = counts
        elif weight_mode == "cosine":
            denominator = np.sqrt(left * right)
            weights = np.divide(
                counts,
                denominator,
                out=np.zeros_like(counts),
                where=denominator > 0,
            )
        else:
            raise ValueError(f"unsupported item-item weight mode: {weight_mode}")

        finite = np.isfinite(weights)
        columns = columns[finite]
        weights = weights[finite]
        if columns.size == 0:
            continue
        if columns.size > top_k:
            selected = np.argpartition(-weights, top_k - 1)[:top_k]
        else:
            selected = np.arange(columns.size)
        # Deterministic tie breaking is part of the reusable execution ABI.
        selected = selected[
            np.lexsort((columns[selected], -weights[selected]))
        ]
        output_rows.extend([row_id] * int(selected.size))
        output_columns.extend(int(value) for value in columns[selected])
        output_values.extend(float(value) for value in weights[selected])

    import scipy.sparse as sp

    shape = (int(cooccurrence.shape[0]), int(cooccurrence.shape[1]))
    return sp.csr_matrix(
        (
            np.asarray(output_values, dtype=np.float32),
            (
                np.asarray(output_rows, dtype=np.int64),
                np.asarray(output_columns, dtype=np.int64),
            ),
        ),
        shape=shape,
        dtype=np.float32,
    )


def _train_interactions_csr(model: Any) -> Any:
    known = getattr(model, "_recclaw_machine_owned_sampler_interactions", None)
    if known is None:
        known = model._recclaw_machine_owned_sampler_dataset.inter_matrix(
            form="csr"
        ).tocsr()
        known.sum_duplicates()
        known.eliminate_zeros()
        model._recclaw_machine_owned_sampler_interactions = known
    return known


def _chunked_legal_score_pool(
    model: Any,
    *,
    user_ids: Any,
    user_vectors: Any,
    item_repr: Any,
    epoch_idx: int,
    negative_count: int,
    replacement: bool,
    score_block: Callable[[Any, Any, Any, int], Any],
    item_chunk_size: int,
    pool_size: int,
    current_positive_ids: Any | None = None,
) -> tuple[Any, Any]:
    """Score only legal item ids and retain a bounded per-user policy pool."""

    import numpy as np
    import torch

    n_items = int(model.n_items)
    retained = min(n_items - 1, max(int(negative_count), int(pool_size)))
    if replacement:
        # Replacement may legitimately request more negatives than there are
        # distinct non-padding items.  Reserve enough columns for the repeated
        # legal ids populated after the full-catalog scan below.
        retained = max(retained, int(negative_count))
    device = item_repr.device
    row_count = int(user_ids.numel())
    running_scores = torch.full(
        (row_count, retained),
        -torch.inf,
        device=device,
        dtype=item_repr.dtype,
    )
    running_items = torch.zeros(
        (row_count, retained),
        device=device,
        dtype=torch.long,
    )
    known = _train_interactions_csr(model)
    cpu_user_ids = user_ids.detach().cpu().reshape(-1).tolist()
    cpu_positive_ids = (
        current_positive_ids.detach().cpu().reshape(-1).tolist()
        if current_positive_ids is not None
        else None
    )
    # Gather the same CSR exclusions once per row chunk. Upload/scatter a whole
    # item block together instead of synchronizing the GPU once per user.
    exclusions = known[cpu_user_ids].tocoo()
    excluded_rows = exclusions.row
    excluded_items = exclusions.col
    if cpu_positive_ids is not None:
        excluded_rows = np.concatenate((excluded_rows, np.arange(row_count)))
        excluded_items = np.concatenate((excluded_items, cpu_positive_ids))

    for item_start in range(1, n_items, int(item_chunk_size)):
        item_end = min(item_start + int(item_chunk_size), n_items)
        item_ids = torch.arange(
            item_start,
            item_end,
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            block = score_block(
                model,
                user_vectors,
                item_repr[item_start:item_end],
                int(epoch_idx),
            )
        if tuple(block.shape) != (row_count, item_end - item_start):
            raise RuntimeError("sampler score hook returned the wrong block shape")
        block = block.detach().clone()
        in_block = (excluded_items >= item_start) & (excluded_items < item_end)
        if in_block.any():
            rows = torch.as_tensor(excluded_rows[in_block], device=device, dtype=torch.long)
            columns = torch.as_tensor(
                excluded_items[in_block] - item_start, device=device, dtype=torch.long
            )
            block[rows, columns] = -torch.inf

        candidate_scores = torch.cat((running_scores, block), dim=1)
        candidate_items = torch.cat(
            (running_items, item_ids.expand(row_count, -1)),
            dim=1,
        )
        running_scores, selected = torch.topk(
            candidate_scores,
            k=retained,
            dim=1,
        )
        running_items = candidate_items.gather(1, selected)

    # ``topk`` retains its zero/-inf initialization when fewer admissible
    # items exist than the requested policy pool.  Replace only that internal
    # padding before the scientific selector sees the pool; selected outputs
    # are never repaired after the fact.
    if row_count == 0:
        return running_items, running_scores
    valid = torch.isfinite(running_scores) & (running_items > 0)
    available = valid.sum(dim=1, keepdim=True)
    minimum_available = int(available.min())
    if minimum_available == 0:
        raise RuntimeError("dynamic sampler has no unobserved item for user")
    if minimum_available < negative_count and not replacement:
        raise RuntimeError(
            "dynamic sampler cannot satisfy negative_count without replacement"
        )
    positions = torch.arange(retained, device=device).expand(row_count, -1)
    # Preserve the exact ordered finite pool, including nonfinite hook outputs
    # and replacement repetition. This does not rescore or change the policy.
    packed_order = torch.where(valid, positions, retained).argsort(dim=1, stable=True)
    extended_available = available.clamp_min(negative_count)
    inside = positions < extended_available
    source_positions = torch.where(inside, positions.remainder(available), 0)
    source_indices = packed_order.gather(1, source_positions)
    running_items = running_items.gather(1, source_indices)
    running_scores = running_scores.gather(1, source_indices).masked_fill(~inside, -torch.inf)
    return running_items, running_scores


def _select_from_legal_pool(
    model: Any,
    *,
    user_ids: Any,
    candidate_ids: Any,
    candidate_scores: Any,
    epoch_idx: int,
    negative_count: int,
    replacement: bool,
    select_from_pool: Callable[[Any, Any, Any, Any, int, int, bool], Any]
    | None,
) -> Any:
    """Apply candidate policy inside the legal pool without output repair."""

    import torch

    machine_candidate_ids = candidate_ids
    with torch.no_grad():
        if select_from_pool is None:
            selected_items = machine_candidate_ids[:, :negative_count]
        else:
            selected_items = select_from_pool(
                model,
                user_ids,
                machine_candidate_ids.clone(),
                candidate_scores.detach(),
                int(epoch_idx),
                int(negative_count),
                bool(replacement),
            )
    if not isinstance(selected_items, torch.Tensor):
        raise RuntimeError("sampler selection hook must return a tensor")
    if (
        selected_items.dtype == torch.bool
        or selected_items.is_floating_point()
        or selected_items.is_complex()
    ):
        raise RuntimeError("sampler selection hook must return integer item ids")
    expected_shape = (int(user_ids.numel()), int(negative_count))
    if tuple(selected_items.shape) != expected_shape:
        raise RuntimeError("sampler selection hook returned the wrong cache shape")
    selected_items = selected_items.to(
        device=machine_candidate_ids.device,
        dtype=torch.long,
    )
    row_membership = (
        selected_items.unsqueeze(-1) == machine_candidate_ids.unsqueeze(1)
    ).any(dim=-1)
    if not bool(row_membership.all()):
        raise RuntimeError("sampler selection emitted an item outside its legal pool")
    if not replacement:
        ordered = selected_items.sort(dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise RuntimeError(
                "sampler selection repeated an item with replacement disabled"
            )
    return selected_items


def build_chunked_dynamic_epoch_cache(
    model: Any,
    *,
    epoch_idx: int,
    negative_count: int,
    replacement: bool,
    representation_provider: Callable[[Any, int], tuple[Any, Any]],
    score_block: Callable[[Any, Any, Any, int], Any],
    select_from_pool: Callable[[Any, Any, Any, Any, int, int, bool], Any]
    | None = None,
    user_chunk_size: int = 128,
    item_chunk_size: int = 1024,
    pool_size: int | None = None,
) -> Any:
    """Build one frozen hard-negative cache with bounded score tensors."""

    import torch

    negative_count = int(negative_count)
    if negative_count < 1:
        raise ValueError("dynamic sampler negative_count must be positive")
    user_chunk_size = max(1, int(user_chunk_size))
    item_chunk_size = max(1, int(item_chunk_size))
    n_users = int(model.n_users)
    n_items = int(model.n_items)
    if n_items <= 1:
        raise RuntimeError("dynamic sampler has no non-padding item")

    with torch.no_grad():
        user_repr, item_repr = representation_provider(model, int(epoch_idx))
    if int(user_repr.shape[0]) != n_users or int(item_repr.shape[0]) != n_items:
        raise RuntimeError("sampler representations violate dataset cardinality")
    user_repr = user_repr.detach()
    item_repr = item_repr.detach()
    device = item_repr.device
    requested_pool = negative_count if pool_size is None else int(pool_size)
    cache = torch.zeros((n_users, negative_count), dtype=torch.long)

    for user_start in range(0, n_users, user_chunk_size):
        user_end = min(user_start + user_chunk_size, n_users)
        user_ids = torch.arange(
            user_start,
            user_end,
            device=device,
            dtype=torch.long,
        )
        candidate_ids, candidate_scores = _chunked_legal_score_pool(
            model,
            user_ids=user_ids,
            user_vectors=user_repr[user_start:user_end],
            item_repr=item_repr,
            epoch_idx=int(epoch_idx),
            negative_count=negative_count,
            replacement=bool(replacement),
            score_block=score_block,
            item_chunk_size=item_chunk_size,
            pool_size=requested_pool,
        )
        selected_items = _select_from_legal_pool(
            model,
            user_ids=user_ids,
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            epoch_idx=int(epoch_idx),
            negative_count=negative_count,
            replacement=bool(replacement),
            select_from_pool=select_from_pool,
        )
        cache[user_start:user_end] = selected_items.detach().cpu()

    model._recclaw_machine_owned_epoch_negative_cache = cache
    return cache


def select_chunked_dynamic_batch_negatives(
    model: Any,
    interaction: Any,
    *,
    epoch_idx: int,
    negative_count: int,
    replacement: bool,
    representation_provider: Callable[[Any, int], tuple[Any, Any]],
    score_block: Callable[[Any, Any, Any, int], Any],
    select_from_pool: Callable[[Any, Any, Any, Any, int, int, bool], Any]
    | None = None,
    user_chunk_size: int = 128,
    item_chunk_size: int = 1024,
    pool_size: int | None = None,
) -> Any:
    """Select batch negatives from the same machine-owned legal score pool."""

    import torch

    negative_count = int(negative_count)
    if negative_count < 1:
        raise ValueError("dynamic sampler negative_count must be positive")
    n_users = int(model.n_users)
    n_items = int(model.n_items)
    if n_items <= 1:
        raise RuntimeError("dynamic sampler has no non-padding item")
    with torch.no_grad():
        user_repr, item_repr = representation_provider(model, int(epoch_idx))
    if int(user_repr.shape[0]) != n_users or int(item_repr.shape[0]) != n_items:
        raise RuntimeError("sampler representations violate dataset cardinality")
    user_repr = user_repr.detach()
    item_repr = item_repr.detach()

    users = interaction[model.USER_ID]
    positives = interaction[model.ITEM_ID]
    if int(users.numel()) != int(positives.numel()):
        raise RuntimeError("sampler interaction user/item shapes do not align")
    flat_users = users.detach().reshape(-1).to(
        device=item_repr.device,
        dtype=torch.long,
    )
    flat_positives = positives.detach().reshape(-1).to(
        device=item_repr.device,
        dtype=torch.long,
    )
    requested_pool = negative_count if pool_size is None else int(pool_size)
    selected_chunks = []
    user_chunk_size = max(1, int(user_chunk_size))
    item_chunk_size = max(1, int(item_chunk_size))
    for row_start in range(0, int(flat_users.numel()), user_chunk_size):
        row_end = min(row_start + user_chunk_size, int(flat_users.numel()))
        user_ids = flat_users[row_start:row_end]
        candidate_ids, candidate_scores = _chunked_legal_score_pool(
            model,
            user_ids=user_ids,
            user_vectors=user_repr[user_ids],
            item_repr=item_repr,
            epoch_idx=int(epoch_idx),
            negative_count=negative_count,
            replacement=bool(replacement),
            score_block=score_block,
            item_chunk_size=item_chunk_size,
            pool_size=requested_pool,
            current_positive_ids=flat_positives[row_start:row_end],
        )
        selected_chunks.append(
            _select_from_legal_pool(
                model,
                user_ids=user_ids,
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                epoch_idx=int(epoch_idx),
                negative_count=negative_count,
                replacement=bool(replacement),
                select_from_pool=select_from_pool,
            )
        )
    if not selected_chunks:
        selected = torch.empty(
            (0, negative_count),
            device=item_repr.device,
            dtype=torch.long,
        )
    else:
        selected = torch.cat(selected_chunks, dim=0)
    if negative_count == 1:
        return selected.reshape(users.shape).to(users.device)
    return selected.reshape((*users.shape, negative_count)).to(users.device)


def lookup_epoch_negative_cache(model: Any, interaction: Any) -> Any:
    """Return the frozen cache rows for the interaction users."""

    cache = getattr(model, "_recclaw_machine_owned_epoch_negative_cache", None)
    if cache is None:
        raise RuntimeError("recclaw_sampler_refresh must run before sampler lookup")
    users = interaction[model.USER_ID]
    selected = cache[users.detach().cpu().long()].to(users.device)
    if selected.shape[-1] == 1:
        return selected.reshape(users.shape)
    return selected.reshape(*users.shape, selected.shape[-1])


__all__ = [
    "build_chunked_dynamic_epoch_cache",
    "build_item_item_cooccurrence_topk",
    "lookup_epoch_negative_cache",
    "select_chunked_dynamic_batch_negatives",
]
