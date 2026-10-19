"""Machine-owned candidate-local decoding for the frozen LIGER profile."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any


_RESOLUTION_PRIMITIVE = "resolution.invalid_sid_drop_lookup"
_RERANK_PRIMITIVE = "retrieval.generated_legal_dense_rerank_to_score"
_DENSE_OBJECTIVE = "objective.train_seen_full_catalog_dense_cross_entropy"
_PARALLEL_LOCAL_DECODE = "decode.parallel_valid_assignment"
_FROZEN_LETTER_TOKENIZER = "tokenizer.frozen_letter_collision_suffix"
_CONSTRAINED_BEAM = "decode.constrained_beam_trie"
_BINDING_ATTRIBUTE = "_recclaw_liger_canonical_decode"


def preserve_liger_context_input_gradients(tree: ast.AST) -> None:
    """Keep learned hook outputs live in the otherwise exact parent forward.

    The content asset is already detached when loaded into its frozen buffer.
    Detaching again after recclaw_prepare_liger_batch freezes learned context.
    """
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name != "model_forward":
            continue
        for statement in ast.walk(function):
            if (
                isinstance(statement, ast.Assign)
                and ast.unparse(statement.value)
                == "batch['input_embeddings'].to(device).detach()"
            ):
                statement.value = statement.value.func.value


class BatchedHostBeamScorer:
    """Keep native beam bookkeeping on CPU, with one transfer per tensor batch.

    Transformers 4.41.2 processes hypotheses in Python. Keeping those scalar
    operations on the GPU synchronizes thousands of times per SID batch. The
    native scorer still owns EOS, score ordering, ties, and finalization.
    """

    def __init__(self, scorer: Any) -> None:
        import torch

        self.scorer = scorer
        scorer.device = torch.device("cpu")
        scorer._done = scorer._done.cpu()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scorer, name)

    @staticmethod
    def _host(value: Any) -> Any:
        import torch

        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, tuple):
            return tuple(BatchedHostBeamScorer._host(item) for item in value)
        return value

    def _call(self, method: Any, input_ids: Any, *args: Any, **kwargs: Any) -> Any:
        result = method(
            input_ids.cpu(),
            *(self._host(value) for value in args),
            **{key: self._host(value) for key, value in kwargs.items()},
        )
        return {
            key: value.to(input_ids.device) if value is not None else None
            for key, value in result.items()
        }

    def process(self, input_ids: Any, *args: Any, **kwargs: Any) -> Any:
        return self._call(self.scorer.process, input_ids, *args, **kwargs)

    def finalize(self, input_ids: Any, *args: Any, **kwargs: Any) -> Any:
        return self._call(self.scorer.finalize, input_ids, *args, **kwargs)


class CatalogPrefixLogitsProcessor:
    """Apply the frozen catalog trie to all beams in one device operation."""

    def __init__(
        self,
        item_sids: Any,
        *,
        vocab_size: int,
        decoder_start_token_id: int,
        eos_token_id: int,
    ) -> None:
        import torch

        rows = item_sids[1:].detach().cpu().tolist()
        self.depth = int(item_sids.shape[1])
        self.radix = vocab_size
        self.decoder_start = decoder_start_token_id
        self.tables = []
        for depth in range(self.depth + 1):
            allowed: dict[int, set[int]] = {}
            for row in rows:
                key = 0
                for token in row[:depth]:
                    key = key * vocab_size + int(token)
                allowed.setdefault(key, set()).add(
                    int(row[depth]) if depth < self.depth else eos_token_id
                )
            keys = sorted(allowed)
            mask = torch.zeros((len(keys), vocab_size), dtype=torch.bool)
            for index, key in enumerate(keys):
                mask[index, list(allowed[key])] = True
            self.tables.append((torch.tensor(keys, dtype=torch.long), mask))
        self.device = None
        self.device_tables = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        depth = int(input_ids.shape[1]) - 1
        if depth < 0 or depth > self.depth:
            raise ValueError("decoder prefix exceeds the frozen SID depth")
        if not bool(torch.all(input_ids[:, 0] == self.decoder_start)):
            raise ValueError("decoder prefix lacks the configured start token")
        if self.device != scores.device:
            self.device_tables = [
                (keys.to(scores.device), mask.to(scores.device))
                for keys, mask in self.tables
            ]
            self.device = scores.device
        keys, allowed = self.device_tables[depth]
        codes = torch.zeros(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        for column in range(1, input_ids.shape[1]):
            codes = codes * self.radix + input_ids[:, column]
        index = torch.searchsorted(keys, codes).clamp_max(keys.numel() - 1)
        if not bool(torch.all(keys[index] == codes)):
            raise ValueError("decoder reached a non-catalog SID prefix")
        mask = torch.zeros_like(scores).masked_fill(~allowed[index], -torch.inf)
        return scores + mask


def machine_owned_catalog_beam(component_specs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Own the declared builtin, not an additional custom decoding mechanism."""
    specs = [spec for spec in component_specs.values() if isinstance(spec, Mapping)]
    decoders = [spec for spec in specs if spec.get("slot_id") == "DECODING_STRATEGY"
                or spec.get("primitive_id") == _CONSTRAINED_BEAM]
    if (len(decoders) != 1 or decoders[0].get("primitive_id") != _CONSTRAINED_BEAM
            or not any(spec.get("primitive_id") == _FROZEN_LETTER_TOKENIZER for spec in specs)):
        return None
    return decoders[0]["parameters"]


def _decode_contract(
    component_specs: Mapping[str, Any],
) -> tuple[int, int, float, bool, bool, bool] | None:
    by_primitive = {
        str(spec.get("primitive_id")): spec
        for spec in component_specs.values()
        if isinstance(spec, Mapping)
    }
    if not {
        _RESOLUTION_PRIMITIVE,
        _RERANK_PRIMITIVE,
        _DENSE_OBJECTIVE,
    }.issubset(by_primitive):
        return None
    rerank = by_primitive[_RERANK_PRIMITIVE]
    objective = by_primitive[_DENSE_OBJECTIVE]
    candidate_width = int(rerank["parameters"]["generator_topk"])
    temperature = float(objective["parameters"]["temperature"])
    position_local_output = {
        _PARALLEL_LOCAL_DECODE,
        _FROZEN_LETTER_TOKENIZER,
    }.issubset(by_primitive)
    return (
        candidate_width, 4, temperature, position_local_output,
        {_CONSTRAINED_BEAM, _FROZEN_LETTER_TOKENIZER}.issubset(by_primitive),
        machine_owned_catalog_beam(component_specs) is not None,
    )


def bind_semantic_id_model_metadata(model_class: type) -> type:
    """Bind the fixed data ABI without choosing the research model's architecture."""
    from recbole.utils import InputType, ModelType

    model_class.type = ModelType.SEQUENTIAL
    model_class.input_type = InputType.POINTWISE
    return model_class


def bind_semantic_id_decode_model_class(
    model_class: type,
    component_specs: Mapping[str, Any],
) -> type:
    """Own resolve/filter/rerank while consuming one candidate generation hook.

    The candidate owns ``recclaw_generation_batch`` and returns its real
    generated SID tuples, the query state trained by its loss, and the exact
    history used to condition generation. The ordinary beam remains
    unconstrained. An explicitly declared catalog-trie beam uses a batched
    processor instead of one host callback and device write for every beam.
    """

    contract = _decode_contract(component_specs)
    if contract is None or getattr(model_class, _BINDING_ATTRIBUTE, None) == contract:
        return model_class
    (
        candidate_width, sid_depth, temperature,
        position_local_output, constrained_beam, builtin_catalog_beam,
    ) = contract
    beam_parameters = machine_owned_catalog_beam(component_specs)

    class _SemanticIdDecodeModel(model_class):
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            import torch

            outputs = super().forward(*args, **kwargs)
            if kwargs.get("labels") is not None:
                # Preserve the exact history mask used by the positive forward,
                # including a declared context hook. Do not run that hook twice.
                outputs.recclaw_encoder_attention_mask = kwargs["attention_mask"]
                device_type = outputs.encoder_last_hidden_state.device.type
                outputs.recclaw_forward_autocast = {
                    "device_type": device_type,
                    "enabled": torch.is_autocast_enabled(device_type),
                    "dtype": torch.get_autocast_dtype(device_type),
                }
            return outputs

        def recclaw_complete_sid_token_log_probs(self, outputs: Any, sids: Any) -> Any:
            """Return log p(code_t | history, code_<t), shaped like [B,(K),4].

            ``outputs`` is the live positive teacher-forced parent output.
            ``sids`` contains catalog-absolute tokens, not item IDs. The caller
            owns sampling, reduction and the additional objective. No encoder
            is rerun and no graph, score or parent query state is detached.
            """
            import torch
            import torch.nn.functional as functional
            from transformers import T5ForConditionalGeneration
            from transformers.modeling_outputs import BaseModelOutput

            if sids.ndim not in (2, 3) or sids.shape[-1] != sid_depth:
                raise ValueError("complete SID scores require [B,4] or [B,K,4] tokens")
            hidden = outputs.encoder_last_hidden_state
            if sids.shape[0] != hidden.shape[0]:
                raise ValueError("complete SID scores must align with encoder histories")
            count = 1 if sids.ndim == 2 else sids.shape[1]
            codes = sids.reshape(-1, sid_depth)
            # The native T5 forward owns shift-right input semantics, tied-head
            # scaling and decoder execution. Bypass TIGER.forward so this extra
            # decoder pass cannot overwrite its dense-retrieval query state.
            with torch.autocast(**outputs.recclaw_forward_autocast):
                decoded = T5ForConditionalGeneration.forward(
                    self,
                    encoder_outputs=BaseModelOutput(
                        last_hidden_state=hidden.repeat_interleave(count, dim=0),
                    ),
                    attention_mask=outputs.recclaw_encoder_attention_mask.repeat_interleave(
                        count, dim=0,
                    ),
                    decoder_input_ids=self._shift_right(codes),
                    use_cache=False,
                    return_dict=True,
                )
            return functional.log_softmax(decoded.logits.float(), dim=-1).gather(
                -1, codes.unsqueeze(-1),
            ).squeeze(-1).reshape(sids.shape)

        def _beam_search(self, input_ids: Any, beam_scorer: Any, *args: Any, **kwargs: Any) -> Any:
            if input_ids.device.type == "cuda":
                beam_scorer = BatchedHostBeamScorer(beam_scorer)
            return super()._beam_search(input_ids, beam_scorer, *args, **kwargs)

        def recclaw_generation_kwargs(self, input_batch: Any) -> dict[str, Any]:
            if builtin_catalog_beam:
                # This registered primitive's complete decoding meaning is known.
                # Do not invoke its old free-form duplicate processor or hook.
                kwargs = {
                    "num_beams": int(beam_parameters["beam_width"]),
                    "num_return_sequences": candidate_width,
                    "max_new_tokens": sid_depth + 1,
                    "length_penalty": float(beam_parameters["length_normalization"]),
                    "use_cache": True,
                }
            else:
                kwargs = super().recclaw_generation_kwargs(input_batch)
            if not constrained_beam:
                return kwargs
            processor = getattr(self, "_recclaw_catalog_prefix_processor", None)
            if processor is None:
                processor = CatalogPrefixLogitsProcessor(
                    self.item_sids,
                    vocab_size=int(self.config.vocab_size),
                    decoder_start_token_id=int(self.config.decoder_start_token_id),
                    eos_token_id=int(self.config.eos_token_id),
                )
                self._recclaw_catalog_prefix_processor = processor
            kwargs = dict(kwargs)
            kwargs.pop("prefix_allowed_tokens_fn", None)
            kwargs["logits_processor"] = [
                *(kwargs.get("logits_processor") or ()), processor
            ]
            return kwargs

        def recclaw_postprocess_generated(self, generated: Any, input_batch: Any) -> Any:
            if not builtin_catalog_beam:
                return super().recclaw_postprocess_generated(generated, input_batch)
            # Native T5 output is start + four SID tokens + terminal EOS.
            return generated[:, 1:1 + sid_depth].reshape(
                input_batch["input_ids"].shape[0], candidate_width, sid_depth
            )

        def _recclaw_catalog_sid_map(self) -> dict[tuple[int, ...], int]:
            item_sids = self.item_sids
            state = (
                int(item_sids.data_ptr()),
                int(item_sids._version),
                tuple(item_sids.shape),
            )
            if getattr(self, "_recclaw_catalog_sid_map_state", None) != state:
                mapping: dict[tuple[int, ...], int] = {}
                collisions: set[tuple[int, ...]] = set()
                for item_id, row in enumerate(
                    item_sids[1:].detach().cpu().tolist(),
                    start=1,
                ):
                    key = tuple(int(token) for token in row)
                    if key in mapping:
                        collisions.add(key)
                    else:
                        mapping[key] = item_id
                for key in collisions:
                    mapping[key] = -1
                self._recclaw_cached_catalog_sid_map = mapping
                self._recclaw_catalog_sid_map_state = state
            return self._recclaw_cached_catalog_sid_map

        def recclaw_resolve_semantic_ids(self, semantic_ids: Any) -> Any:
            import torch

            if not isinstance(semantic_ids, torch.Tensor) or (
                semantic_ids.ndim != 3
                or semantic_ids.shape[1] != candidate_width
                or semantic_ids.shape[2] != sid_depth
            ):
                raise ValueError(
                    "generated semantic IDs must have shape "
                    f"[batch,{candidate_width},{sid_depth}]"
                )
            if position_local_output:
                from .sid_codec_scaffold import absolute_frozen_letter_sids

                semantic_ids = absolute_frozen_letter_sids(
                    semantic_ids,
                    codes_per_level=256,
                    depth=sid_depth,
                )
            sid_map = self._recclaw_catalog_sid_map()
            resolved = [
                sid_map.get(tuple(int(token) for token in row), -1)
                for row in semantic_ids.detach().cpu().reshape(-1, sid_depth).tolist()
            ]
            return torch.tensor(
                resolved,
                dtype=torch.long,
                device=semantic_ids.device,
            ).reshape(semantic_ids.shape[:2])

        def _recclaw_catalog_query_scores(
            self,
            query_state: Any,
            generated_items: Any,
        ) -> Any:
            import torch
            import torch.nn.functional as F

            if not isinstance(query_state, torch.Tensor) or (
                query_state.ndim != 2 or query_state.shape[-1] != 128
            ):
                raise ValueError("generation query state must have shape [batch,128]")
            parameters = tuple(self.emb_proj.parameters())
            cache_state = (
                int(self.item_content.data_ptr()),
                int(self.item_content._version),
                tuple(int(parameter._version) for parameter in parameters),
            )
            if getattr(self, "_recclaw_catalog_projection_state", None) != cache_state:
                catalog = self.emb_proj(self.item_content[1:].float())
                self._recclaw_cached_catalog_projection = F.normalize(
                    catalog,
                    dim=-1,
                ).detach()
                self._recclaw_catalog_projection_state = cache_state
            catalog = self._recclaw_cached_catalog_projection.to(
                device=query_state.device,
                dtype=query_state.dtype,
            )
            safe_items = generated_items.clamp(min=1, max=self.n_items - 1) - 1
            generated_vectors = catalog[safe_items]
            query = F.normalize(query_state, dim=-1)
            return (query.unsqueeze(1) * generated_vectors).sum(-1) / temperature

        def _recclaw_generated_rows(self, interaction: Any, targets: Any) -> tuple[Any, ...]:
            import torch

            generated = super().recclaw_generation_batch(interaction, targets)
            if not isinstance(generated, tuple) or len(generated) != 3:
                raise ValueError(
                    "recclaw_generation_batch must return "
                    "(semantic_ids, query_state, history)"
                )
            semantic_ids, query_state, history = generated
            batch_size = int(targets.shape[0])
            if not isinstance(history, torch.Tensor) or (
                history.ndim != 2 or history.shape[0] != batch_size
            ):
                raise ValueError("generation history must have shape [batch,length]")
            if semantic_ids.shape[0] != batch_size or query_state.shape[0] != batch_size:
                raise ValueError("generation outputs must preserve the external batch")
            generated_items = self.recclaw_resolve_semantic_ids(semantic_ids)
            candidate_scores = self._recclaw_catalog_query_scores(
                query_state,
                generated_items,
            )
            return generated_items, candidate_scores, history

        @staticmethod
        def _recclaw_legal_positions(
            generated_items: Any,
            history: Any,
            targets: Any,
        ) -> list[list[int]]:
            generated_rows = generated_items.detach().cpu().tolist()
            history_rows = history.detach().cpu().tolist()
            target_rows = targets.detach().cpu().tolist()
            rows: list[list[int]] = []
            for generated_row, history_row, target_value in zip(
                generated_rows,
                history_rows,
                target_rows,
                strict=True,
            ):
                target = int(target_value)
                seen = {
                    int(item)
                    for item in history_row
                    if item > 0
                }
                accepted: set[int] = set()
                positions: list[int] = []
                for position, value in enumerate(generated_row):
                    item = int(value)
                    if (
                        item < 1
                        or (item in seen and item != target)
                        or item in accepted
                    ):
                        continue
                    accepted.add(item)
                    positions.append(position)
                rows.append(positions)
            return rows

        def recclaw_target_ranks(self, interaction: Any, targets: Any) -> list[int | None]:
            import torch

            generated_items, candidate_scores, history = self._recclaw_generated_rows(
                interaction,
                targets,
            )
            positions = self._recclaw_legal_positions(
                generated_items,
                history,
                targets,
            )
            ranks: list[int | None] = []
            for row_index, legal_positions in enumerate(positions):
                if not legal_positions:
                    ranks.append(None)
                    continue
                source = candidate_scores.new_tensor(
                    legal_positions,
                    dtype=torch.long,
                )
                order = torch.argsort(
                    candidate_scores[row_index, source],
                    descending=True,
                    stable=True,
                )[:10]
                ranked = generated_items[row_index, source[order]].detach().cpu().tolist()
                target = int(targets[row_index])
                ranks.append(ranked.index(target) if target in ranked else None)
            return ranks

        def _hybrid_scores(self, interaction: Any) -> Any:
            import torch

            targets = (
                interaction[self.POS_ITEM_ID]
                if self.POS_ITEM_ID in interaction
                else torch.ones(
                    len(interaction),
                    dtype=torch.long,
                    device=interaction[self.ITEM_SEQ].device,
                )
            )
            generated_items, candidate_scores, history = self._recclaw_generated_rows(
                interaction,
                targets,
            )
            positions = self._recclaw_legal_positions(
                generated_items,
                history,
                targets,
            )
            floor = torch.finfo(candidate_scores.dtype).min
            scores = candidate_scores.new_full(
                (candidate_scores.shape[0], self.n_items),
                floor,
            )
            for row_index, legal_positions in enumerate(positions):
                if not legal_positions:
                    continue
                source = torch.tensor(
                    legal_positions,
                    dtype=torch.long,
                    device=candidate_scores.device,
                )
                items = generated_items[row_index, source]
                scores[row_index, items] = candidate_scores[row_index, source]
            return scores

        def predict(self, interaction: Any) -> Any:
            scores = self._hybrid_scores(interaction)
            return scores.gather(
                1,
                interaction[self.ITEM_ID].view(-1, 1),
            ).squeeze(1)

        def full_sort_predict(self, interaction: Any) -> Any:
            return self._hybrid_scores(interaction).reshape(-1)

    setattr(_SemanticIdDecodeModel, _BINDING_ATTRIBUTE, contract)
    _SemanticIdDecodeModel.__name__ = model_class.__name__
    _SemanticIdDecodeModel.__qualname__ = model_class.__qualname__
    _SemanticIdDecodeModel.__module__ = model_class.__module__
    return _SemanticIdDecodeModel


__all__ = ["CatalogPrefixLogitsProcessor", "bind_semantic_id_decode_model_class"]
