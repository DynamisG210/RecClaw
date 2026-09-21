"""Machine-owned RecBole ABI and lifecycle binding for compiled candidates.

The compiler owns stable hook names, legal item enumeration, and epoch-cache
lifecycle.  Candidate code owns the scientific scoring and selection policy
through the implementation hooks captured when the class is bound.  The
standard ``sampler.uniform`` primitive is fully mechanical, so its frozen epoch
cache is constructed here rather than delegated to the Implementer.
"""

from __future__ import annotations

from functools import wraps
from inspect import signature
from math import gcd
from typing import Any, Mapping

from recclaw_core.mechanism_space.spectral_selection import spectral_index_ranges

from .compiled_efficiency_kernels import (
    build_chunked_dynamic_epoch_cache,
    build_item_item_cooccurrence_topk,
    lookup_epoch_negative_cache,
    select_chunked_dynamic_batch_negatives,
)


_MASK64 = (1 << 64) - 1
_TRAINER_REFRESH_GUARD = "_recclaw_machine_owned_trainer_refresh_epoch"
_TRAIN_DATA_FIT_GUARD = "_recclaw_machine_owned_train_data_fit_in_progress"
_TRAIN_DATA_FIT_CALLS = "_recclaw_machine_owned_train_data_fit_hook_calls"
_TRAIN_DATA_FIT_COMPLETE = "_recclaw_machine_owned_train_data_fit_complete"
_TRAIN_DATA_FIT_ROLES = "_recclaw_machine_owned_train_data_fit_roles"
POST_DEVICE_MECHANISM_INIT_FLAG = "__recclaw_post_device_mechanism_init__"
_POST_DEVICE_MECHANISM_CONFIG = "_recclaw_deferred_mechanism_config"
_POST_DEVICE_MECHANISM_INIT_COMPLETE = (
    "_recclaw_post_device_mechanism_init_complete"
)
_TRAIN_ITEM_GRAPH_ROLE = "TRAIN_ITEM_GRAPH"
_TRAIN_SPECTRAL_BASIS_ROLE = "TRAIN_SPECTRAL_BASIS"
_TRAIN_STATISTICS_ROLE = "TRAIN_STATISTICS"
_SUPPORTED_TRAIN_DATA_FIT_ROLES = frozenset(
    {_TRAIN_ITEM_GRAPH_ROLE, _TRAIN_SPECTRAL_BASIS_ROLE, _TRAIN_STATISTICS_ROLE}
)


class CandidateCardinalityContractError(RuntimeError):
    """Candidate-owned constructor state violates the RecBole dataset ABI."""

    implicated_methods = ("__init__",)
    implicated_files = ("recclaw_ext/candidate.py",)

    def __init__(
        self,
        message: str,
        *,
        implicated_methods: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        if implicated_methods is not None:
            self.implicated_methods = implicated_methods


class TrainSpectralBasisContractError(RuntimeError):
    """A compiled train-only graph or spectral contract is not executable.

    The public name is retained because callers already use it as the stable
    failure type for the machine-owned train-data fit boundary.
    """

    implicated_methods = ("recclaw_fit_operator",)
    implicated_files = ("recclaw_ext/candidate.py",)

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class TrainSpectralBasisLifecycleError(TrainSpectralBasisContractError):
    """The machine-owned pre-optimization fit lifecycle was bypassed."""


def required_train_data_fit_roles(
    component_specs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return supported train-derived DATA roles that require one pre-fit."""

    roles = {
        str(source.get("data_role"))
        for spec in component_specs.values()
        if isinstance(spec, Mapping)
        for item in spec.get("inputs", ())
        if isinstance(item, Mapping)
        and isinstance(item.get("source"), Mapping)
        for source in (item["source"],)
        if source.get("kind") == "DATA"
        and source.get("data_role") in _SUPPORTED_TRAIN_DATA_FIT_ROLES
    }
    return tuple(sorted(roles))


def train_spectral_basis_rank(
    component_specs: Mapping[str, Any],
) -> int | None:
    """Resolve the one typed rank shared by all spectral-basis consumers."""

    ranks: list[int] = []
    for spec in component_specs.values():
        if not isinstance(spec, Mapping):
            continue
        consumes_basis = any(
            isinstance(item, Mapping)
            and isinstance(item.get("source"), Mapping)
            and item["source"].get("kind") == "DATA"
            and item["source"].get("data_role") == _TRAIN_SPECTRAL_BASIS_ROLE
            for item in spec.get("inputs", ())
        )
        if not consumes_basis:
            continue
        parameters = spec.get("parameters")
        rank = parameters.get("rank") if isinstance(parameters, Mapping) else None
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise TrainSpectralBasisContractError(
                "TRAIN_SPECTRAL_BASIS_RANK_INVALID",
                "every TRAIN_SPECTRAL_BASIS consumer requires one positive "
                "integer rank",
            )
        ranks.append(rank)
    if not ranks:
        return None
    if len(set(ranks)) != 1:
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_RANK_CONFLICT",
            "all consumers of one TRAIN_SPECTRAL_BASIS must declare the same rank",
        )
    return ranks[0]


def train_spectral_basis_ranges(
    component_specs: Mapping[str, Any], *, rank: int,
) -> tuple[tuple[int, int], ...]:
    """One shared DATA basis cannot denote different column selections."""
    selections = set()
    for spec in component_specs.values():
        if not isinstance(spec, Mapping):
            continue
        if not any(
            item.get("source", {}).get("kind") == "DATA"
            and item["source"].get("data_role") == _TRAIN_SPECTRAL_BASIS_ROLE
            for item in spec.get("inputs", ())
        ):
            continue
        parameters = spec["parameters"]
        ranges = parameters.get("spectral_index_ranges", ((0, rank),))
        if spec.get("primitive_id") == "state.spectral_graph_coordinates":
            ranges = parameters.get("spectral_index_ranges")
            if "frequency_band" in parameters:
                raise TrainSpectralBasisContractError(
                    "TRAIN_SPECTRAL_BASIS_SELECTION_INVALID",
                    "frequency_band has no executable mapping; revise the new draft",
                )
        try:
            selections.add(spectral_index_ranges(rank, ranges))
        except ValueError as error:
            raise TrainSpectralBasisContractError(
                "TRAIN_SPECTRAL_BASIS_SELECTION_INVALID", str(error),
            ) from error
    if len(selections) > 1:
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_SELECTION_CONFLICT",
            "all consumers of one TRAIN_SPECTRAL_BASIS must select the same columns",
        )
    return next(iter(selections))


def build_train_spectral_basis(
    train_dataset: Any, *, rank: int,
    index_ranges: tuple[tuple[int, int], ...] | None = None,
) -> Any:
    """Build a deterministic item spectral subspace from train interactions."""

    import numpy as np
    import torch
    from scipy import sparse
    from scipy.sparse.linalg import LinearOperator, eigsh

    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_RANK_INVALID",
            "spectral rank must be a positive integer",
        )
    try:
        ranges = spectral_index_ranges(
            rank, ((0, rank),) if index_ranges is None else index_ranges,
        )
    except ValueError as error:
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_SELECTION_INVALID", str(error),
        ) from error
    depth = ranges[-1][1]
    matrix_builder = getattr(train_dataset, "inter_matrix", None)
    if not callable(matrix_builder):
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_DATASET_INVALID",
            "TRAIN_SPECTRAL_BASIS requires the RecBole train Dataset inter_matrix ABI",
        )
    interactions = matrix_builder(form="csr")
    if not sparse.issparse(interactions) or len(interactions.shape) != 2:
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_DATASET_INVALID",
            "train Dataset inter_matrix must return a two-dimensional sparse matrix",
        )
    interactions = interactions.tocsr().astype(np.float64, copy=True)
    if interactions.shape[1] <= 1:
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_CARDINALITY_INVALID",
            "TRAIN_SPECTRAL_BASIS requires at least one non-padding item",
        )
    interactions = interactions[1:, 1:]
    interactions.data.fill(1.0)
    interactions.eliminate_zeros()
    maximum_rank = min(interactions.shape) - 1
    if depth > maximum_rank:
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_RANK_INFEASIBLE",
            "required spectral depth must be smaller than both train-user and "
            "non-padding-item cardinality",
        )
    user_gram = LinearOperator(
        shape=(interactions.shape[0], interactions.shape[0]),
        matvec=lambda vector: interactions @ (interactions.T @ vector),
        matmat=lambda matrix: interactions @ (interactions.T @ matrix),
        dtype=np.dtype(np.float64),
    )
    eigenvalues, user_vectors = eigsh(
        user_gram,
        k=depth,
        which="LA",
        # A non-symmetric deterministic start keeps the generic solver
        # independent of campaign seeds while spanning ordinary eigenspaces.
        v0=(
            np.sin(np.arange(1, interactions.shape[0] + 1) * np.sqrt(2.0))
            + np.cos(np.arange(1, interactions.shape[0] + 1) * np.sqrt(3.0))
        ),
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    user_vectors = user_vectors[:, order]
    tolerance = np.finfo(np.float64).eps * max(interactions.shape) * max(
        1.0,
        float(eigenvalues[0]),
    )
    if float(eigenvalues[-1]) <= tolerance:
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_RANK_INFEASIBLE",
            "required spectral depth exceeds the numerical rank of train interactions",
        )
    pivots = np.argmax(np.abs(user_vectors), axis=0)
    signs = np.sign(user_vectors[pivots, np.arange(depth)])
    signs[signs == 0.0] = 1.0
    user_vectors *= signs
    if depth != rank:
        indices = [index for start, stop in ranges for index in range(start, stop)]
        user_vectors = user_vectors[:, indices]
        eigenvalues = eigenvalues[indices]
    basis = interactions.T @ (
        user_vectors / np.sqrt(eigenvalues).reshape(1, -1)
    )
    padded = np.zeros((interactions.shape[1] + 1, rank), dtype=np.float32)
    padded[1:, :] = basis.astype(np.float32, copy=False)
    if not np.isfinite(padded).all():
        raise TrainSpectralBasisContractError(
            "TRAIN_SPECTRAL_BASIS_NONFINITE",
            "train-derived spectral basis contains nonfinite values",
        )
    return torch.from_numpy(padded)


def build_train_item_graph(train_dataset: Any) -> Any:
    """Build the raw sparse train-only item co-occurrence graph.

    Sparse enumeration, padding exclusion, self-edge removal, and deterministic
    ordering remain owned by the existing compiler kernel.  Using the complete
    non-padding catalog as ``top_k`` preserves every observed off-diagonal edge;
    normalization and propagation remain mechanism variables for the candidate.
    """

    import numpy as np
    import torch
    from scipy import sparse

    item_num = getattr(train_dataset, "item_num", None)
    if (
        not isinstance(item_num, int)
        or isinstance(item_num, bool)
        or item_num <= 1
    ):
        raise TrainSpectralBasisContractError(
            "TRAIN_ITEM_GRAPH_CARDINALITY_INVALID",
            "TRAIN_ITEM_GRAPH requires at least one non-padding item",
        )
    matrix_builder = getattr(train_dataset, "inter_matrix", None)
    if not callable(matrix_builder):
        raise TrainSpectralBasisContractError(
            "TRAIN_ITEM_GRAPH_DATASET_INVALID",
            "TRAIN_ITEM_GRAPH requires the RecBole train Dataset inter_matrix ABI",
        )
    graph = build_item_item_cooccurrence_topk(
        train_dataset,
        top_k=item_num - 1,
        weight_mode="count",
    )
    if not sparse.issparse(graph) or graph.shape != (item_num, item_num):
        raise TrainSpectralBasisContractError(
            "TRAIN_ITEM_GRAPH_DATASET_INVALID",
            "compiler-owned item graph must match the train Dataset item cardinality",
        )
    graph = graph.tocoo(copy=False)
    if not np.isfinite(graph.data).all():
        raise TrainSpectralBasisContractError(
            "TRAIN_ITEM_GRAPH_NONFINITE",
            "train-derived item graph contains nonfinite values",
        )
    indices = torch.from_numpy(
        np.vstack((graph.row, graph.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(graph.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(item_num, item_num),
        dtype=torch.float32,
    ).coalesce()


def prepare_candidate_train_data(model: Any, train_data: Any) -> None:
    """Run the candidate train-data hook once before trainer optimization."""

    if (
        getattr(model, POST_DEVICE_MECHANISM_INIT_FLAG, False) is True
        and not getattr(model, _POST_DEVICE_MECHANISM_INIT_COMPLETE, False)
    ):
        initializer = getattr(model, "recclaw_initialize_mechanism")
        mechanism_config = getattr(model, _POST_DEVICE_MECHANISM_CONFIG)
        initializer(
            mechanism_config,
            train_data._dataset,
        )
        model.to(mechanism_config["device"])
        setattr(model, _POST_DEVICE_MECHANISM_INIT_COMPLETE, True)
        setattr(model, "_recclaw_machine_owned_mechanism_initialized_v1", True)

    fit_operator = getattr(model, "recclaw_fit_operator", None)
    required_roles = tuple(getattr(model, _TRAIN_DATA_FIT_ROLES, ()))
    if not required_roles:
        if callable(fit_operator):
            fit_operator(train_data)
        return
    if not callable(fit_operator):
        raise TrainSpectralBasisLifecycleError(
            "TRAIN_DATA_FIT_HOOK_MISSING",
            "a declared train-derived input requires recclaw_fit_operator",
        )
    calls = int(getattr(model, _TRAIN_DATA_FIT_CALLS, 0))
    if calls != 0:
        raise TrainSpectralBasisLifecycleError(
            "TRAIN_DATA_FIT_CALL_COUNT_INVALID",
            "train-derived inputs must be fitted exactly once before optimization",
        )
    setattr(model, _TRAIN_DATA_FIT_GUARD, True)
    try:
        fit_operator(train_data)
    finally:
        setattr(model, _TRAIN_DATA_FIT_GUARD, False)
    if int(getattr(model, _TRAIN_DATA_FIT_CALLS, 0)) != 1:
        raise TrainSpectralBasisLifecycleError(
            "TRAIN_DATA_FIT_CALL_COUNT_INVALID",
            "train-derived fit hook did not complete exactly once",
        )


def _bind_dataset_cardinality_aliases(dataset: Any) -> tuple[int, int]:
    """Expose the one RecBole cardinality ABI before candidate construction."""

    user_num = int(getattr(dataset, "user_num"))
    item_num = int(getattr(dataset, "item_num"))
    # Provider code repeatedly guessed Dataset.n_users/n_items even though
    # RecBole owns user_num/item_num.  These aliases are mechanical facts, not
    # candidate state or a mechanism hook, and make the constructor boundary
    # deterministic without teaching each generated model another API dialect.
    setattr(dataset, "n_users", user_num)
    setattr(dataset, "n_items", item_num)
    return user_num, item_num


def _validate_candidate_cardinality(
    model: Any,
    *,
    expected_users: int,
    expected_items: int,
    implicated_methods: tuple[str, ...] = ("__init__",),
) -> None:
    observed_users = getattr(model, "n_users", expected_users)
    observed_items = getattr(model, "n_items", expected_items)
    if int(observed_users) != expected_users or int(observed_items) != expected_items:
        raise CandidateCardinalityContractError(
            "candidate cardinalities drifted from the RecBole Dataset ABI",
            implicated_methods=implicated_methods,
        )
    for field_name, expected in (
        ("user_embedding", expected_users),
        ("item_embedding", expected_items),
    ):
        embedding = getattr(model, field_name, None)
        observed = getattr(embedding, "num_embeddings", expected)
        if int(observed) != expected:
            raise CandidateCardinalityContractError(
                f"candidate {field_name} cardinality drifted from the "
                f"RecBole Dataset ABI: expected {expected}, observed {observed}",
                implicated_methods=implicated_methods,
            )


def _epoch_sampler_specs(
    component_specs: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        spec
        for spec in component_specs.values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "NEGATIVE_SAMPLER"
        and isinstance(spec.get("parameters"), Mapping)
        and spec["parameters"].get("refresh_frequency") == "EPOCH"
    )


def _sampler_specs(
    component_specs: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        spec
        for spec in component_specs.values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "NEGATIVE_SAMPLER"
        and isinstance(spec.get("parameters"), Mapping)
    )


def _primitive_specs(
    component_specs: Mapping[str, Any],
    primitive_id: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        spec
        for spec in component_specs.values()
        if isinstance(spec, Mapping) and spec.get("primitive_id") == primitive_id
    )


def _owns_canonical_mf_dot_product_full_sort(
    component_specs: Mapping[str, Any],
) -> bool:
    """Return whether the compiler fully determines the evaluation scorer."""

    score_specs = tuple(
        spec
        for spec in component_specs.values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "SCORE_HEAD"
        and spec.get("primitive_id") == "score.dot_product"
    )
    if len(score_specs) != 1:
        return False

    by_id = {
        str(spec.get("component_id")): spec
        for spec in component_specs.values()
        if isinstance(spec, Mapping) and spec.get("component_id")
    }
    score_inputs = score_specs[0].get("inputs", ())
    if not isinstance(score_inputs, (list, tuple)) or len(score_inputs) != 1:
        return False
    score_source = score_inputs[0].get("source") if isinstance(score_inputs[0], Mapping) else None
    if not isinstance(score_source, Mapping) or score_source.get("kind") != "COMPONENT":
        return False
    encoder = by_id.get(str(score_source.get("component_id", "")))
    if (
        not isinstance(encoder, Mapping)
        or encoder.get("slot_id") != "ENCODER"
        or encoder.get("primitive_id") != "encoder.none_mf"
        or encoder.get("custom_component_id") is not None
    ):
        return False
    encoder_inputs = encoder.get("inputs", ())
    if not isinstance(encoder_inputs, (list, tuple)) or not encoder_inputs:
        return False
    table_primitives = {"embedding.independent_user_item"}
    for input_spec in encoder_inputs:
        source = input_spec.get("source") if isinstance(input_spec, Mapping) else None
        if not isinstance(source, Mapping) or source.get("kind") != "COMPONENT":
            return False
        embedding = by_id.get(str(source.get("component_id", "")))
        if (
            not isinstance(embedding, Mapping)
            or embedding.get("slot_id") != "EMBEDDING"
            or embedding.get("primitive_id") not in table_primitives
            or embedding.get("custom_component_id") is not None
        ):
            return False
        embedding_inputs = embedding.get("inputs", ())
        if not isinstance(embedding_inputs, (list, tuple)):
            return False
        for embedding_input in embedding_inputs:
            embedding_source = (
                embedding_input.get("source")
                if isinstance(embedding_input, Mapping)
                else None
            )
            if (
                not isinstance(embedding_source, Mapping)
                or embedding_source.get("kind") != "DATA"
                or embedding_source.get("data_role") not in {"USER_ID", "ITEM_ID"}
            ):
                return False
    return True


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _config_integer(config: Any, key: str, default: int) -> int:
    try:
        value = config[key]
    except Exception:
        value = default
    return int(default if value is None else value)


def _known_positive_items(model: Any) -> tuple[frozenset[int], ...]:
    dataset = model._recclaw_machine_owned_sampler_dataset
    inter_matrix = getattr(dataset, "inter_matrix", None)
    if callable(inter_matrix):
        matrix = inter_matrix(form="csr").tocsr()
        positives: list[frozenset[int]] = []
        for user_id in range(int(model.n_users)):
            start = int(matrix.indptr[user_id])
            end = int(matrix.indptr[user_id + 1])
            positives.append(
                frozenset(int(item_id) for item_id in matrix.indices[start:end])
            )
        return tuple(positives)

    # Small qualification fixtures may expose the same training truth through
    # inter_feat only.  Keep this fallback aligned with the API-contract
    # validator instead of trusting candidate-owned positive dictionaries.
    positives_by_user: list[set[int]] = [
        set() for _ in range(int(model.n_users))
    ]
    inter_feat = getattr(dataset, "inter_feat", None)
    if inter_feat is None:
        return tuple(frozenset() for _ in positives_by_user)
    users = inter_feat[model.USER_ID]
    items = inter_feat[model.ITEM_ID]
    if hasattr(users, "detach"):
        users = users.detach().cpu().reshape(-1).tolist()
    if hasattr(items, "detach"):
        items = items.detach().cpu().reshape(-1).tolist()
    for user_id, item_id in zip(users, items):
        user_id = int(user_id)
        item_id = int(item_id)
        if 0 <= user_id < len(positives_by_user) and item_id > 0:
            positives_by_user[user_id].add(item_id)
    return tuple(frozenset(items) for items in positives_by_user)


def _uniform_epoch_refresh(model: Any, epoch_idx: int) -> None:
    import torch

    spec = model._recclaw_machine_owned_uniform_sampler_spec
    parameters = spec["parameters"]
    negative_count = int(parameters.get("negative_count", 1))
    replacement = bool(parameters.get("replacement", True))
    if negative_count < 1:
        raise RuntimeError("sampler.uniform negative_count must be positive")

    known_positives = model._recclaw_machine_owned_known_positives
    cache = torch.empty(
        (int(model.n_users), negative_count),
        dtype=torch.long,
    )
    for user_id in range(int(model.n_users)):
        allowed = tuple(
            item_id
            for item_id in range(1, int(model.n_items))
            if item_id not in known_positives[user_id]
        )
        if not allowed:
            raise RuntimeError(
                f"sampler.uniform has no unobserved item for user {user_id}"
            )
        if not replacement and negative_count > len(allowed):
            raise RuntimeError(
                "sampler.uniform cannot sample the declared count without replacement"
            )
        key = (
            int(model._recclaw_machine_owned_sampler_seed)
            ^ ((int(epoch_idx) + 1) * 0xD1B54A32D192ED03)
            ^ ((user_id + 1) * 0x94D049BB133111EB)
        ) & _MASK64
        offset = _splitmix64(key) % len(allowed)
        if replacement:
            chosen = [
                allowed[_splitmix64(key + index) % len(allowed)]
                for index in range(negative_count)
            ]
        else:
            stride = int(_splitmix64(key ^ 0xA0761D6478BD642F) % len(allowed)) or 1
            while gcd(stride, len(allowed)) != 1:
                stride += 1
            chosen = [
                allowed[(offset + index * stride) % len(allowed)]
                for index in range(negative_count)
            ]
        cache[user_id] = torch.tensor(chosen, dtype=torch.long)
    model._recclaw_machine_owned_epoch_negative_cache = cache


def _uniform_epoch_step(model: Any, interaction: Any) -> Any:
    cache = getattr(model, "_recclaw_machine_owned_epoch_negative_cache", None)
    if cache is None:
        raise RuntimeError(
            "recclaw_sampler_refresh must run before recclaw_sampler_step"
        )
    users = interaction[model.USER_ID]
    negatives = cache[users.detach().cpu().long()].to(users.device)
    if negatives.shape[-1] == 1:
        return negatives.reshape(users.shape)
    return negatives.reshape(*users.shape, negatives.shape[-1])


def bind_epoch_sampler_model_class(
    candidate_class: type[Any],
    component_specs: Mapping[str, Any],
) -> type[Any]:
    """Bind the stable RecBole ABI while leaving mechanism policy LLM-owned."""

    train_data_fit_roles = required_train_data_fit_roles(component_specs)
    spectral_basis_rank = train_spectral_basis_rank(component_specs)
    spectral_basis_ranges = (
        train_spectral_basis_ranges(component_specs, rank=spectral_basis_rank)
        if spectral_basis_rank is not None else None
    )
    sampler_specs = _sampler_specs(component_specs)
    epoch_specs = _epoch_sampler_specs(component_specs)
    item_cooccurrence_specs = _primitive_specs(
        component_specs,
        "relation.item_item_cooccurrence_topk",
    )
    graph_precompute_specs = _primitive_specs(
        component_specs,
        "efficiency.graph_filter_precomputation",
    )
    scored_false_negative_specs = tuple(
        spec
        for spec in sampler_specs
        if spec.get("primitive_id") == "sampler.false_negative_aware"
        and spec["parameters"].get("hardness") in {"DYNAMIC", "CURRICULUM"}
        and spec["parameters"].get("refresh_frequency") in {"BATCH", "EPOCH"}
    )
    score_primitives = {
        str(spec.get("primitive_id"))
        for spec in component_specs.values()
        if isinstance(spec, Mapping) and spec.get("slot_id") == "SCORE_HEAD"
    }
    canonical_mf_full_sort = _owns_canonical_mf_dot_product_full_sort(
        component_specs
    )
    if getattr(candidate_class, "_recclaw_machine_owned_compiled_model_v2", False):
        return candidate_class

    implementation_refresh = getattr(candidate_class, "recclaw_sampler_refresh", None)
    implementation_step = getattr(candidate_class, "recclaw_sampler_step", None)
    implementation_relation_weight = getattr(
        candidate_class,
        "recclaw_item_item_cooccurrence_weight",
        None,
    )
    implementation_sampler_representations = getattr(
        candidate_class,
        "recclaw_sampler_representations",
        None,
    )
    implementation_sampler_score_block = getattr(
        candidate_class,
        "recclaw_sampler_score_block",
        None,
    )
    implementation_sampler_select = getattr(
        candidate_class,
        "recclaw_sampler_select_from_pool",
        None,
    )
    uniform_parameters = (
        sampler_specs[0].get("parameters") if len(sampler_specs) == 1 else None
    )
    uniform_spec = (
        sampler_specs[0]
        if len(sampler_specs) == 1
        and sampler_specs[0].get("primitive_id") == "sampler.uniform"
        and isinstance(uniform_parameters, Mapping)
        and uniform_parameters.get("hardness") in {"NONE", "STATIC"}
        and uniform_parameters.get("false_negative_policy") == "SUPPRESS_KNOWN"
        and (
            uniform_parameters.get("refresh_frequency") == "EPOCH"
            or (
                uniform_parameters.get("refresh_frequency") == "BATCH"
                and int(uniform_parameters.get("negative_count", 1)) == 1
                and bool(uniform_parameters.get("replacement", True))
            )
        )
        else None
    )

    original_init = candidate_class.__init__
    implementation_initialize = getattr(
        candidate_class,
        "recclaw_initialize_mechanism",
        None,
    )
    implementation_full_sort_predict = getattr(
        candidate_class, "full_sort_predict", None
    )

    owns_item_relation_kernel = (
        len(item_cooccurrence_specs) == 1 and len(graph_precompute_specs) == 1
    )
    scored_epoch_false_negative_spec = (
        scored_false_negative_specs[0]
        if len(scored_false_negative_specs) == 1
        and scored_false_negative_specs[0]["parameters"].get(
            "refresh_frequency"
        )
        == "EPOCH"
        else None
    )
    scored_batch_false_negative_spec = (
        scored_false_negative_specs[0]
        if len(scored_false_negative_specs) == 1
        and scored_false_negative_specs[0]["parameters"].get(
            "refresh_frequency"
        )
        == "BATCH"
        else None
    )

    def machine_owned_item_relation(self: Any, dataset: Any) -> Any:
        parameters = item_cooccurrence_specs[0].get("parameters", {})
        top_k = int(parameters.get("top_k", getattr(self, "top_k", 64)))
        weight_transform = None
        if callable(implementation_relation_weight):
            weight_transform = lambda counts, left, right: implementation_relation_weight(
                self,
                counts,
                left,
                right,
            )
        return build_item_item_cooccurrence_topk(
            dataset,
            top_k=top_k,
            weight_mode="cosine",
            weight_transform=weight_transform,
        )

    def machine_owned_sampler_representations(
        self: Any,
        epoch_idx: int,
    ) -> tuple[Any, Any]:
        if callable(implementation_sampler_representations):
            result = implementation_sampler_representations(self, int(epoch_idx))
        else:
            raise RuntimeError(
                "dynamic sampler requires recclaw_sampler_representations"
            )
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError(
                "recclaw_sampler_representations must return user/item tensors"
            )
        return result[0], result[1]

    def machine_owned_sampler_score_block(
        self: Any,
        user_block: Any,
        item_block: Any,
        epoch_idx: int,
    ) -> Any:
        if callable(implementation_sampler_score_block):
            return implementation_sampler_score_block(
                self,
                user_block,
                item_block,
                int(epoch_idx),
            )
        if "score.normalized_dot_product" in score_primitives:
            import torch.nn.functional as functional

            user_block = functional.normalize(user_block, dim=-1)
            item_block = functional.normalize(item_block, dim=-1)
        if score_primitives & {"score.dot_product", "score.normalized_dot_product"}:
            return user_block @ item_block.transpose(0, 1)
        raise RuntimeError("dynamic sampler requires recclaw_sampler_score_block")

    def machine_owned_scored_sampler_options(
        self: Any,
        spec: Mapping[str, Any],
    ) -> tuple[int, bool, Any, int]:
        parameters = spec["parameters"]
        hardness = parameters.get("hardness")
        false_negative_policy = parameters.get("false_negative_policy")
        needs_policy_hook = (
            hardness == "CURRICULUM"
            or false_negative_policy != "SUPPRESS_KNOWN"
        )
        if needs_policy_hook and not (
            callable(implementation_sampler_score_block)
            or callable(implementation_sampler_select)
        ):
            raise RuntimeError(
                "CURRICULUM or policy-aware sampler requires a scientific "
                "recclaw_sampler_score_block or recclaw_sampler_select_from_pool"
            )
        negative_count = int(parameters.get("negative_count", 1))
        selector = (
            implementation_sampler_select
            if callable(implementation_sampler_select)
            else None
        )
        if selector is None:
            pool_size = negative_count
        else:
            raw_pool_size = getattr(self, "recclaw_sampler_pool_size", None)
            if raw_pool_size is None:
                raise RuntimeError(
                    "recclaw_sampler_select_from_pool requires an explicit "
                    "recclaw_sampler_pool_size"
                )
            pool_size = int(raw_pool_size)
            if pool_size < negative_count:
                raise RuntimeError(
                    "recclaw_sampler_pool_size cannot be smaller than negative_count"
                )
        return (
            negative_count,
            bool(parameters.get("replacement", True)),
            selector,
            pool_size,
        )

    def machine_owned_dynamic_refresh(self: Any, epoch_idx: int) -> Any:
        if scored_epoch_false_negative_spec is None:
            raise RuntimeError("no machine-owned scored EPOCH sampler is declared")
        negative_count, replacement, selector, pool_size = (
            machine_owned_scored_sampler_options(
                self,
                scored_epoch_false_negative_spec,
            )
        )
        return build_chunked_dynamic_epoch_cache(
            self,
            epoch_idx=int(epoch_idx),
            negative_count=negative_count,
            replacement=replacement,
            representation_provider=machine_owned_sampler_representations,
            score_block=machine_owned_sampler_score_block,
            select_from_pool=selector,
            user_chunk_size=int(
                getattr(self, "recclaw_sampler_user_chunk_size", 128)
            ),
            item_chunk_size=int(
                getattr(self, "recclaw_sampler_item_chunk_size", 1024)
            ),
            pool_size=pool_size,
        )

    def machine_owned_dynamic_batch(self: Any, interaction: Any) -> Any:
        if scored_batch_false_negative_spec is None:
            raise RuntimeError("no machine-owned scored BATCH sampler is declared")
        negative_count, replacement, selector, pool_size = (
            machine_owned_scored_sampler_options(
                self,
                scored_batch_false_negative_spec,
            )
        )
        return select_chunked_dynamic_batch_negatives(
            self,
            interaction,
            epoch_idx=int(self._recclaw_machine_owned_sampler_epoch),
            negative_count=negative_count,
            replacement=replacement,
            representation_provider=machine_owned_sampler_representations,
            score_block=machine_owned_sampler_score_block,
            select_from_pool=selector,
            user_chunk_size=int(
                getattr(self, "recclaw_sampler_user_chunk_size", 128)
            ),
            item_chunk_size=int(
                getattr(self, "recclaw_sampler_item_chunk_size", 1024)
            ),
            pool_size=pool_size,
        )

    def machine_owned_fit_operator(self: Any, train_data: Any) -> Any:
        if not train_data_fit_roles:
            raise TrainSpectralBasisLifecycleError(
                "TRAIN_DATA_FIT_ROLE_UNSUPPORTED",
                "no machine-owned train-derived input is declared",
            )
        if not bool(getattr(self, _TRAIN_DATA_FIT_GUARD, False)):
            raise TrainSpectralBasisLifecycleError(
                "TRAIN_DATA_FIT_OUTSIDE_LIFECYCLE",
                "recclaw_fit_operator is owned by prepare_candidate_train_data",
            )
        if int(getattr(self, _TRAIN_DATA_FIT_CALLS, 0)) != 0:
            raise TrainSpectralBasisLifecycleError(
                "TRAIN_DATA_FIT_CALL_COUNT_INVALID",
                "train-derived inputs cannot be fitted more than once",
            )
        train_dataset = getattr(train_data, "_dataset", None)
        if train_dataset is None or train_dataset is not getattr(
            self,
            "_recclaw_machine_owned_constructor_dataset",
            None,
        ):
            raise TrainSpectralBasisContractError(
                "TRAIN_DATA_FIT_PROVENANCE_INVALID",
                "fit must consume the exact train Dataset used to construct "
                "the model",
            )
        requires_item_graph = _TRAIN_ITEM_GRAPH_ROLE in train_data_fit_roles
        requires_spectral_basis = (
            _TRAIN_SPECTRAL_BASIS_ROLE in train_data_fit_roles
        )
        requires_statistics = _TRAIN_STATISTICS_ROLE in train_data_fit_roles
        if requires_item_graph and hasattr(self, "recclaw_item_graph"):
            raise TrainSpectralBasisContractError(
                "TRAIN_ITEM_GRAPH_PREBOUND",
                "TRAIN_ITEM_GRAPH must be derived from train interactions "
                "during machine-owned fit",
            )
        if requires_spectral_basis and hasattr(self, "recclaw_spectral_basis"):
            raise TrainSpectralBasisContractError(
                "TRAIN_SPECTRAL_BASIS_PREBOUND",
                "TRAIN_SPECTRAL_BASIS cannot be constructed from item positions "
                "before train-data fit",
            )
        register_buffer = getattr(self, "register_buffer", None)
        if not callable(register_buffer):
            reason_code = (
                "TRAIN_SPECTRAL_BASIS_MODEL_INVALID"
                if train_data_fit_roles == (_TRAIN_SPECTRAL_BASIS_ROLE,)
                else "TRAIN_DATA_FIT_MODEL_INVALID"
            )
            raise TrainSpectralBasisContractError(
                reason_code,
                "compiled train-derived candidates must expose "
                "torch.nn.Module.register_buffer",
            )

        item_graph = (
            build_train_item_graph(train_dataset) if requires_item_graph else None
        )
        basis = (
            build_train_spectral_basis(
                train_dataset, rank=spectral_basis_rank,
                index_ranges=spectral_basis_ranges,
            )
            if requires_spectral_basis and spectral_basis_rank is not None
            else None
        )
        item_popularity = None
        if requires_statistics:
            import numpy as np
            import torch

            matrix = train_dataset.inter_matrix(form="csr")
            item_popularity = torch.from_numpy(
                np.asarray(matrix.sum(axis=0)).reshape(-1).copy()
            ).float()
            item_popularity[0] = 0.0
        device = getattr(self, "device", None)
        if device is not None:
            if item_graph is not None:
                item_graph = item_graph.to(device=device)
            if basis is not None:
                basis = basis.to(device=device)
            if item_popularity is not None:
                item_popularity = item_popularity.to(device=device)
        if item_graph is not None:
            register_buffer("recclaw_item_graph", item_graph)
        if basis is not None:
            register_buffer("recclaw_spectral_basis", basis)
        if item_popularity is not None:
            register_buffer("recclaw_item_popularity", item_popularity)
        setattr(self, _TRAIN_DATA_FIT_CALLS, 1)
        setattr(self, _TRAIN_DATA_FIT_COMPLETE, True)
        provenance: dict[str, Any] = {
            "data_roles": train_data_fit_roles,
            "dataset_identity": id(train_dataset),
            "item_cardinality": int(getattr(train_dataset, "item_num")),
        }
        if len(train_data_fit_roles) == 1:
            provenance["data_role"] = train_data_fit_roles[0]
        if item_graph is not None:
            provenance["item_graph_edge_count"] = int(item_graph._nnz())
        if basis is not None:
            provenance["rank"] = int(basis.shape[1])
            provenance["spectral_index_ranges"] = spectral_basis_ranges
            provenance["spectral_depth"] = spectral_basis_ranges[-1][1]
        self._recclaw_machine_owned_train_data_fit_provenance = provenance
        if basis is not None and item_graph is None and item_popularity is None:
            return basis
        if item_graph is not None and basis is None and item_popularity is None:
            return item_graph
        if train_data_fit_roles == (_TRAIN_STATISTICS_ROLE,):
            return item_popularity
        fitted = {
            _TRAIN_ITEM_GRAPH_ROLE: item_graph,
            _TRAIN_SPECTRAL_BASIS_ROLE: basis,
        }
        if item_popularity is not None:
            fitted[_TRAIN_STATISTICS_ROLE] = item_popularity
        return fitted

    @wraps(original_init)
    def machine_owned_init(
        self: Any,
        config: Any,
        dataset: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        expected_users, expected_items = _bind_dataset_cardinality_aliases(dataset)
        object.__setattr__(self, _TRAIN_DATA_FIT_ROLES, train_data_fit_roles)
        object.__setattr__(self, _TRAIN_DATA_FIT_CALLS, 0)
        object.__setattr__(self, _TRAIN_DATA_FIT_COMPLETE, False)
        object.__setattr__(self, _TRAIN_DATA_FIT_GUARD, False)
        if spectral_basis_rank is not None:
            object.__setattr__(self, "recclaw_rank", spectral_basis_rank)
        # GeneralRecommender does not retain the Dataset object, while compiled
        # candidates legitimately need the same framework-owned dataset during
        # their own constructor (for example, to precompute a graph).  Bind the
        # stable constructor ABI before entering candidate code, then restore
        # the exact object after construction in case candidate code overwrote
        # the attribute.
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(
            self,
            "_recclaw_machine_owned_constructor_dataset",
            dataset,
        )
        original_init(self, config, dataset, *args, **kwargs)
        self.dataset = dataset
        _validate_candidate_cardinality(
            self,
            expected_users=expected_users,
            expected_items=expected_items,
        )
        self.n_users = expected_users
        self.n_items = expected_items
        self._recclaw_machine_owned_constructor_dataset = dataset
        self._recclaw_machine_owned_padding_item_id = 0
        self._recclaw_machine_owned_sampler_dataset = dataset
        self._recclaw_machine_owned_sampler_seed = _config_integer(config, "seed", 0)
        if uniform_spec is not None:
            self._recclaw_machine_owned_known_positives = _known_positive_items(self)
        if (
            uniform_spec is not None
            and uniform_spec["parameters"].get("refresh_frequency") == "EPOCH"
        ):
            self._recclaw_machine_owned_uniform_sampler_spec = uniform_spec
            self._recclaw_machine_owned_epoch_negative_cache = None
        if scored_false_negative_specs:
            self._recclaw_machine_owned_sampler_epoch = 0
        if (
            callable(implementation_initialize)
            and getattr(self, POST_DEVICE_MECHANISM_INIT_FLAG, False) is not True
            and not getattr(
                self,
                "_recclaw_machine_owned_mechanism_initialized_v1",
                False,
            )
        ):
            self._recclaw_machine_owned_mechanism_initialized_v1 = True
            if len(signature(implementation_initialize).parameters) == 1:
                implementation_initialize(self)
            else:
                implementation_initialize(self, config, dataset)
        if spectral_basis_rank is not None:
            observed_rank = getattr(self, "recclaw_rank", spectral_basis_rank)
            if (
                not isinstance(observed_rank, int)
                or isinstance(observed_rank, bool)
                or observed_rank != spectral_basis_rank
            ):
                raise TrainSpectralBasisContractError(
                    "TRAIN_SPECTRAL_BASIS_RANK_MISMATCH",
                    "candidate spectral width does not match the compiled rank",
                )
            self.recclaw_rank = spectral_basis_rank
            if hasattr(self, "recclaw_spectral_basis"):
                raise TrainSpectralBasisContractError(
                    "TRAIN_SPECTRAL_BASIS_PREBOUND",
                    "TRAIN_SPECTRAL_BASIS must be derived from train interactions "
                    "during machine-owned fit",
                )
        if (
            _TRAIN_ITEM_GRAPH_ROLE in train_data_fit_roles
            and hasattr(self, "recclaw_item_graph")
        ):
            raise TrainSpectralBasisContractError(
                "TRAIN_ITEM_GRAPH_PREBOUND",
                "TRAIN_ITEM_GRAPH must be derived from train interactions "
                "during machine-owned fit",
            )
        _validate_candidate_cardinality(
            self,
            expected_users=expected_users,
            expected_items=expected_items,
            implicated_methods=("recclaw_initialize_mechanism",),
        )

    def machine_owned_refresh(self: Any, epoch_idx: int) -> Any:
        if scored_false_negative_specs:
            self._recclaw_machine_owned_sampler_epoch = int(epoch_idx)
        if getattr(self, _TRAINER_REFRESH_GUARD, object()) == int(epoch_idx):
            return None
        if scored_epoch_false_negative_spec is not None:
            return machine_owned_dynamic_refresh(self, int(epoch_idx))
        if scored_batch_false_negative_spec is not None:
            return None
        if uniform_spec is not None:
            return _uniform_epoch_refresh(self, int(epoch_idx))
        if not callable(implementation_refresh):
            raise RuntimeError(
                "EPOCH sampler mechanism must provide a cache builder"
            )
        return implementation_refresh(self, int(epoch_idx))

    def machine_owned_step(self: Any, interaction: Any) -> Any:
        if scored_epoch_false_negative_spec is not None:
            # This cache was constructed exclusively from the compiler-owned
            # legal pool, so returning it directly preserves the candidate's
            # selection policy and avoids a second fallback-style rewrite.
            return lookup_epoch_negative_cache(self, interaction)
        if scored_batch_false_negative_spec is not None:
            return machine_owned_dynamic_batch(self, interaction)
        if uniform_spec is not None:
            if uniform_spec["parameters"].get("refresh_frequency") == "BATCH":
                return interaction[self.NEG_ITEM_ID]
            return _uniform_epoch_step(self, interaction)
        if not callable(implementation_step):
            raise RuntimeError(
                "EPOCH sampler mechanism must provide a frozen-cache lookup"
            )
        return implementation_step(self, interaction)

    if owns_item_relation_kernel:
        candidate_class.recclaw_build_item_item_cooccurrence_topk = (
            machine_owned_item_relation
        )
        # Private candidate methods have no compiler-owned signature or
        # weighting contract. Only the explicit recclaw_* ABI is replaceable.
    if scored_epoch_false_negative_spec is not None:
        candidate_class.recclaw_build_dynamic_epoch_cache = (
            machine_owned_dynamic_refresh
        )
    candidate_class.__init__ = machine_owned_init
    if train_data_fit_roles:
        candidate_class.recclaw_fit_operator = machine_owned_fit_operator
    if epoch_specs or scored_batch_false_negative_spec is not None:
        candidate_class.recclaw_sampler_refresh = machine_owned_refresh
    if (
        epoch_specs
        or uniform_spec is not None
        or scored_batch_false_negative_spec is not None
    ):
        candidate_class.recclaw_sampler_step = machine_owned_step

    if callable(implementation_full_sort_predict):
        @wraps(implementation_full_sort_predict)
        def machine_owned_full_sort_predict(
            self: Any,
            interaction: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if canonical_mf_full_sort:
                import torch

                users = interaction[self.USER_ID].to(
                    device=self.item_embedding.weight.device,
                    dtype=torch.long,
                )
                result = self.user_embedding(users) @ self.item_embedding.weight.t()
                result = result.clone()
                padding_item_id = int(
                    self._recclaw_machine_owned_padding_item_id
                )
                if 0 <= padding_item_id < int(self.n_items):
                    result[..., padding_item_id] = result.new_tensor(
                        torch.finfo(result.dtype).min
                    )
            else:
                result = implementation_full_sort_predict(
                    self,
                    interaction,
                    *args,
                    **kwargs,
                )
            batch_size = getattr(interaction, "length", None)
            if batch_size is None:
                batch_size = next(iter(interaction.values())).shape[0]
            expected = int(batch_size) * int(self.n_items)
            observed = int(result.numel()) if hasattr(result, "numel") else -1
            if observed != expected:
                raise RuntimeError(
                    "full_sort_predict must preserve exactly one column for every "
                    "RecBole item id including padding: "
                    f"expected {expected}, observed {observed}"
                )
            return result

        candidate_class.full_sort_predict = machine_owned_full_sort_predict

    if uniform_spec is not None:
        original_calculate_loss = candidate_class.calculate_loss

        @wraps(original_calculate_loss)
        def machine_owned_calculate_loss(self: Any, interaction: Any) -> Any:
            sampled = self.recclaw_sampler_step(interaction)
            original_negative = interaction[self.NEG_ITEM_ID]
            interaction[self.NEG_ITEM_ID] = sampled
            try:
                return original_calculate_loss(self, interaction)
            finally:
                interaction[self.NEG_ITEM_ID] = original_negative

        candidate_class.calculate_loss = machine_owned_calculate_loss

    if train_data_fit_roles:
        if train_data_fit_roles == (_TRAIN_SPECTRAL_BASIS_ROLE,):
            missing_fit_reason = "TRAIN_SPECTRAL_BASIS_NOT_FITTED"
            missing_fit_message = (
                "train-derived spectral basis must be fitted before model execution"
            )
        elif train_data_fit_roles == (_TRAIN_ITEM_GRAPH_ROLE,):
            missing_fit_reason = "TRAIN_ITEM_GRAPH_NOT_FITTED"
            missing_fit_message = (
                "train-derived item graph must be fitted before model execution"
            )
        else:
            missing_fit_reason = "TRAIN_DATA_FIT_NOT_COMPLETED"
            missing_fit_message = (
                "all declared train-derived inputs must be fitted before model execution"
            )

        def guard_train_data_fit(method: Any) -> Any:
            @wraps(method)
            def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
                if not bool(getattr(self, _TRAIN_DATA_FIT_COMPLETE, False)):
                    raise TrainSpectralBasisLifecycleError(
                        missing_fit_reason,
                        missing_fit_message,
                    )
                return method(self, *args, **kwargs)

            return guarded

        for method_name in ("calculate_loss", "predict", "full_sort_predict"):
            method = getattr(candidate_class, method_name, None)
            if callable(method):
                setattr(candidate_class, method_name, guard_train_data_fit(method))

    candidate_class._recclaw_machine_owned_epoch_model_v1 = bool(
        epoch_specs or scored_batch_false_negative_spec is not None
    )
    candidate_class._recclaw_machine_owned_compiled_model_v2 = True
    candidate_class._recclaw_machine_owned_train_data_fit_roles_v1 = (
        train_data_fit_roles
    )
    return candidate_class


def bind_epoch_sampler_trainer_class(trainer_class: type[Any]) -> type[Any]:
    """Refresh the bound model once at the start of every training epoch."""

    if getattr(trainer_class, "_recclaw_machine_owned_epoch_trainer_v1", False):
        return trainer_class
    implementation_train_epoch = trainer_class._train_epoch

    @wraps(implementation_train_epoch)
    def machine_owned_train_epoch(
        self: Any,
        train_data: Any,
        epoch_idx: int,
        loss_func: Any = None,
        show_progress: bool = False,
    ) -> Any:
        self.model.recclaw_sampler_refresh(epoch_idx)
        setattr(self.model, _TRAINER_REFRESH_GUARD, int(epoch_idx))
        try:
            return implementation_train_epoch(
                self,
                train_data,
                epoch_idx,
                loss_func=loss_func,
                show_progress=show_progress,
            )
        finally:
            delattr(self.model, _TRAINER_REFRESH_GUARD)

    trainer_class._train_epoch = machine_owned_train_epoch
    trainer_class._recclaw_machine_owned_epoch_trainer_v1 = True
    return trainer_class


def bind_train_data_fit_trainer_class(trainer_class: type[Any]) -> type[Any]:
    """Require one completed train-data fit before optimizer construction."""

    if getattr(trainer_class, "_recclaw_machine_owned_train_data_trainer_v1", False):
        return trainer_class
    implementation_init = trainer_class.__init__

    @wraps(implementation_init)
    def machine_owned_init(
        self: Any,
        config: Any,
        model: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        required_roles = tuple(getattr(model, _TRAIN_DATA_FIT_ROLES, ()))
        calls = int(getattr(model, _TRAIN_DATA_FIT_CALLS, 0))
        completed = bool(getattr(model, _TRAIN_DATA_FIT_COMPLETE, False))
        if required_roles and (calls != 1 or not completed):
            raise TrainSpectralBasisLifecycleError(
                "TRAIN_DATA_FIT_CALL_COUNT_INVALID",
                "one train-derived fit must complete before trainer optimizer "
                "construction",
            )
        implementation_init(self, config, model, *args, **kwargs)

    trainer_class.__init__ = machine_owned_init
    trainer_class._recclaw_machine_owned_train_data_trainer_v1 = True
    return trainer_class


__all__ = [
    "POST_DEVICE_MECHANISM_INIT_FLAG",
    "TrainSpectralBasisContractError",
    "TrainSpectralBasisLifecycleError",
    "bind_train_data_fit_trainer_class",
    "bind_epoch_sampler_model_class",
    "bind_epoch_sampler_trainer_class",
    "build_train_item_graph",
    "build_train_spectral_basis",
    "prepare_candidate_train_data",
    "required_train_data_fit_roles",
    "train_spectral_basis_rank",
]
