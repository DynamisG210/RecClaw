"""Direct, content-bound execution of the pinned pre-Research-Line RecClaw.

The decision logic in this module is not a rewrite of Original RecClaw.  The
release loader materializes the exact Git blobs from the pinned Main commit and
imports its ``RecClawAgent``.  The adapter only projects common BL executable
actions into the Original registry contract and projects its decisions back.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest
from .contracts import validate_no_research_evidence_authority_fields


ORIGINAL_MAIN_COMMIT = "2d8c881354e1b536a6c66d7dfbb977e0c5090e50"
ORIGINAL_MAIN_FILES: Mapping[str, tuple[str, str]] = {
    "configs/action_space.yaml": (
        "0fd22af7d917218dac6d63b0aef35fbec20a4e37",
        "e45b93d6118412ed3d324fdd0d50c617a78f36e3973acb6aedb2f4c4d88e0f24",
    ),
    "configs/candidate_proposal_schema.yaml": (
        "8310207ba08edfd503df14455442970f2d03ced1",
        "a1fca4c546bc945b6a6ee5acc406a166a855bc0fb7355a1e5cd148a0e9b826a6",
    ),
    "scripts/action_space.py": (
        "12109fa4942cc2d5892f890464615361f602ddc8",
        "b9befe128d0aa89a5fa892a16478fafc184cca3f2b3e28a0b7b1cd9d7b367e78",
    ),
    "scripts/agent.py": (
        "c40334b72dbb9557eced2bd081915b5333156fdf",
        "2c73c25fc9212a27a506d465e5aaca1606e7d729c827c5f7e67a0984b926b6d0",
    ),
    "scripts/collect_result.py": (
        "481a80b1828f0fa7ade2fa68a03ab2b6c7815f0d",
        "92d71c159a29b8d9541296d483928db271e73d3ada3275f9ca53ed8d2f63c255",
    ),
    "scripts/compare_runs.py": (
        "267d5a86bc4464708deff499e667c758f07a7ca2",
        "b4a6f237323b15c47a683280e9cdc245fcc1b61cd30f2d79bff0b756cd214698",
    ),
}


class OriginalMainSourceError(RuntimeError):
    pass


class _ProposalRefreshRequested(RuntimeError):
    pass


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _load_file_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OriginalMainSourceError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class OriginalMainSourceReleaseV1:
    """Materialize and import exact immutable Main blobs from the local repo."""

    repository_root: Path
    materialization_root: Path
    _agent_module: ModuleType | None = field(default=None, init=False)

    @property
    def identity_digest(self) -> str:
        return sha256_digest(
            {
                "commit": ORIGINAL_MAIN_COMMIT,
                "files": {
                    path: {"blob_sha1": identity[0], "sha256": identity[1]}
                    for path, identity in sorted(ORIGINAL_MAIN_FILES.items())
                },
                "release": "OriginalMainSourceReleaseV1",
            }
        )

    def materialize(self) -> None:
        for relative, (blob_sha1, expected_sha256) in ORIGINAL_MAIN_FILES.items():
            bound = subprocess.run(
                ["git", "rev-parse", f"{ORIGINAL_MAIN_COMMIT}:{relative}"],
                cwd=self.repository_root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if (
                bound.returncode != 0
                or bound.stdout.strip() != blob_sha1
            ):
                raise OriginalMainSourceError(
                    f"pinned Main path/blob mismatch: {relative}"
                )
            completed = subprocess.run(
                ["git", "cat-file", "blob", blob_sha1],
                cwd=self.repository_root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode != 0:
                raise OriginalMainSourceError(
                    f"pinned Original blob unavailable: {relative}"
                )
            payload = completed.stdout
            if (
                _git_blob_sha1(payload) != blob_sha1
                or hashlib.sha256(payload).hexdigest() != expected_sha256
            ):
                raise OriginalMainSourceError(
                    f"pinned Original blob identity mismatch: {relative}"
                )
            target = self.materialization_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    def load_agent_module(self) -> ModuleType:
        if self._agent_module is not None:
            return self._agent_module
        self.materialize()
        scripts = self.materialization_root / "scripts"
        dependency_names = ("action_space", "collect_result", "compare_runs")
        prior_modules = {
            name: sys.modules.get(name) for name in dependency_names
        }
        loaded_names: list[str] = []
        try:
            for name in dependency_names:
                if name == "compare_runs":
                    continue
                sys.modules.pop(name, None)
                _load_file_module(name, scripts / f"{name}.py")
                loaded_names.append(name)
            sys.modules.pop("compare_runs", None)
            _load_file_module("compare_runs", scripts / "compare_runs.py")
            loaded_names.append("compare_runs")
            module_name = (
                "_recclaw_original_main_"
                + self.identity_digest[:16]
                + "_agent"
            )
            self._agent_module = _load_file_module(
                module_name,
                scripts / "agent.py",
            )
        finally:
            for name in loaded_names:
                sys.modules.pop(name, None)
            for name, module in prior_modules.items():
                if module is not None:
                    sys.modules[name] = module
        return self._agent_module

    def new_agent(self, *, search_seed: int, proposal_every: int) -> Any:
        module = self.load_agent_module()
        config = module.AgentConfig(
            seed=int(search_seed),
            proposal_every=int(proposal_every),
            allow_llm_fallback=False,
        )
        return module.RecClawAgent(config)


@dataclass(slots=True)
class PinnedOriginalMainAdapterV1:
    """Thin BL projection around the exact pinned ``RecClawAgent``."""

    repository_root: Path
    search_seed: int
    proposal_every: int = 3
    _temporary_root: tempfile.TemporaryDirectory[str] = field(init=False)
    _source_release: OriginalMainSourceReleaseV1 = field(init=False)
    _agent_module: ModuleType = field(init=False)
    _agent: Any = field(init=False)
    cached_proposals: tuple[Mapping[str, Any], ...] = field(
        default=(),
        init=False,
    )
    last_refresh_round: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.repository_root = Path(self.repository_root).resolve()
        self._temporary_root = tempfile.TemporaryDirectory(
            prefix="recclaw-original-main-"
        )
        materialization_root = Path(self._temporary_root.name)
        self._source_release = OriginalMainSourceReleaseV1(
            repository_root=self.repository_root,
            materialization_root=materialization_root,
        )
        self._agent_module = self._source_release.load_agent_module()
        self._agent = self._source_release.new_agent(
            search_seed=self.search_seed,
            proposal_every=self.proposal_every,
        )

    @property
    def identity_digest(self) -> str:
        return sha256_digest(
            {
                "adapter": "PinnedOriginalMainAdapterV1",
                "projection": "COMMON_BL_ACTION_TO_ORIGINAL_REGISTRY_V1",
                "source_release_digest": self._source_release.identity_digest,
            }
        )

    @property
    def source_release_digest(self) -> str:
        return self._source_release.identity_digest

    def refresh_required(self, round_index: int) -> bool:
        def request_refresh() -> dict[str, Any]:
            raise _ProposalRefreshRequested

        self._agent.generate_llm_candidate_proposals = request_refresh
        self._agent._load_candidate_proposals = lambda: []
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self._agent.refresh_candidate_proposals(int(round_index))
        except _ProposalRefreshRequested:
            return True
        return False

    def install_proposals(
        self,
        *,
        round_index: int,
        proposals: Sequence[Mapping[str, Any]],
    ) -> None:
        if not proposals:
            raise OriginalMainSourceError("Original proposal refresh is empty")
        self.cached_proposals = tuple(dict(item) for item in proposals)
        self.last_refresh_round = int(round_index)
        proposal_path = Path(self._agent.config.proposal_path)
        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        proposal_path.touch()
        self._agent.force_proposal_refresh = False

    def load_original_state(
        self,
        *,
        memory: Sequence[Mapping[str, Any]] = (),
        last_planner_action: Mapping[str, Any] | None = None,
        force_proposal_refresh: bool = False,
    ) -> None:
        """Load only state fields consumed by the exact Original planner."""

        self._agent.memory = [dict(item) for item in memory]
        self._agent.history_by_candidate = {}
        for item in self._agent.memory:
            if item.get("event"):
                continue
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id:
                self._agent.history_by_candidate.setdefault(
                    candidate_id,
                    [],
                ).append(item)
        self._agent.last_planner_action = dict(last_planner_action or {})
        self._agent.force_proposal_refresh = bool(force_proposal_refresh)

    def set_planner_config(self, **updates: Any) -> None:
        for name, value in updates.items():
            if not hasattr(self._agent.config, name):
                raise OriginalMainSourceError(
                    f"unknown pinned Original config field: {name}"
                )
            setattr(self._agent.config, name, value)

    @staticmethod
    def _registry_candidate(action: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "base_model",
            "candidate_id",
            "entrypoint",
            "priority",
            "runner_type",
            "status",
        }
        missing = sorted(required - set(action))
        if missing:
            raise OriginalMainSourceError(
                "BL projection lacks Original registry fields: "
                + ",".join(missing)
            )
        validate_no_research_evidence_authority_fields(action)
        return {
            **dict(action),
            "consumes": list(action.get("consumes", ())),
            "wired": True,
        }

    def rank(
        self,
        common_eligible_actions: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        if not common_eligible_actions:
            return ()
        candidates = [
            self._registry_candidate(action)
            for action in common_eligible_actions
            if action.get("original_projection_kind", "REGISTRY")
            == "REGISTRY"
        ]
        by_id = {
            str(action["candidate_id"]): dict(action)
            for action in common_eligible_actions
        }
        self._agent.registry = [dict(item) for item in candidates]
        prior_signatures = set(self._agent.scheduled_execution_signatures)
        prior_candidates = set(self._agent.scheduled_candidate_ids)
        prior_proposals = list(self._agent.candidate_proposals)
        prior_report = dict(self._agent.proposal_validation_report)
        accepted_actions = [
            dict(action)
            for action in common_eligible_actions
            if action.get("original_projection_kind")
            == "ACCEPTED_PROPOSAL"
        ]
        self._agent.candidate_proposals = [
            {
                **action,
                "parameter_overrides": dict(
                    action.get("parameter_overrides", {})
                ),
                "proposal_type": str(
                    action.get("proposal_type") or "tuning"
                ),
                "runnable_level": str(
                    action.get("runnable_level") or "config_only"
                ),
            }
            for action in accepted_actions
        ]
        self._agent.proposal_validation_report = {
            "results": [
                {
                    "candidate_id": action["candidate_id"],
                    "parameter_signature": str(
                        action.get("parameter_signature") or ""
                    ),
                    "status": str(
                        action.get("proposal_validation_status")
                        or "accepted"
                    ),
                }
                for action in accepted_actions
            ]
        }
        ordered: list[Mapping[str, Any]] = []
        try:
            while len(ordered) < len(common_eligible_actions):
                chosen, _params, _context = self._agent.plan()
                candidate_id = str(chosen["candidate_id"])
                if any(
                    str(item["candidate_id"]) == candidate_id
                    for item in ordered
                ):
                    break
                ordered.append(by_id[candidate_id])
                execution_signature = str(
                    chosen.get("execution_signature") or ""
                )
                if execution_signature:
                    self._agent.scheduled_execution_signatures.add(
                        execution_signature
                    )
                self._agent.scheduled_candidate_ids.add(candidate_id)
        finally:
            self._agent.scheduled_execution_signatures = prior_signatures
            self._agent.scheduled_candidate_ids = prior_candidates
            self._agent.candidate_proposals = prior_proposals
            self._agent.proposal_validation_report = prior_report
        return tuple(ordered)

    def close_round(self, feedback: Mapping[str, Any]) -> Mapping[str, Any]:
        validate_no_research_evidence_authority_fields(feedback)
        candidate_id = str(feedback["candidate_id"])
        candidate = next(
            (
                item
                for item in self._agent.registry
                if str(item.get("candidate_id")) == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise OriginalMainSourceError(
                "Original feedback candidate is absent from its exact registry"
            )
        outcome = dict(feedback["search_outcome"])
        metrics = dict(outcome.get("normalized_metrics", {}))
        run_status = str(outcome.get("run_status") or "")
        candidate_result = {
            **metrics,
            "status": (
                "success"
                if run_status in {"SUCCESS", "COMPLETED", "SMOKE_PASS"}
                else "crash"
            ),
        }
        compare_baseline = dict(
            feedback.get(
                "compare_baseline",
                {
                    "delta": None,
                    "decision": "crash",
                    "explanation": "typed comparator delta is NOT_AVAILABLE",
                },
            )
        )
        compare_history = dict(
            feedback.get(
                "compare_history_best",
                compare_baseline,
            )
        )
        action_out = {
            "exit_code": 0 if candidate_result["status"] == "success" else 1
        }
        decision, reason, next_action = self._agent.reflect(
            candidate,
            {},
            action_out,
            candidate_result,
            compare_baseline,
            compare_history,
        )
        record = self._agent_module.TrialRecord(
            round_id=int(feedback["round_index"]),
            candidate_id=candidate_id,
            params={},
            run_id=str(feedback.get("run_id") or ""),
            status=candidate_result["status"],
            result=candidate_result,
            compare_baseline=compare_baseline,
            compare_history_best=compare_history,
            dimension_report={},
            decision=decision,
            reason=reason,
            next_action=next_action,
            parent_candidate_id=str(
                candidate.get("parent_candidate_id") or ""
            ),
            execution_signature=str(
                candidate.get("execution_signature") or ""
            ),
        )
        self._agent.remember(record)
        transition = {
            "applied_transition_class": (
                "PINNED_ORIGINAL_MAIN_FEEDBACK_CONSUMED"
            ),
            "decision": decision,
            "feedback_consumption_count": 1,
            "round_feedback_digest": sha256_digest(feedback),
            "search_memory_commit": "NO_WRITE",
            "source_release_digest": self.source_release_digest,
            "state_projection": self.state_projection(),
        }
        return {
            **transition,
            "transition_digest": sha256_digest(transition),
        }

    def state_projection(self) -> Mapping[str, Any]:
        trials = [
            row for row in self._agent.memory if not row.get("event")
        ]
        return canonical_value(
            {
                "executed": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "decision": row.get("decision"),
                        "execution_signature": row.get(
                            "execution_signature"
                        ),
                        "result": row.get("result"),
                    }
                    for row in trials
                ],
                "force_proposal_refresh": bool(
                    self._agent.force_proposal_refresh
                ),
                "last_planner_action": dict(
                    self._agent.last_planner_action
                ),
                "last_refresh_round": self.last_refresh_round,
                "proposal_every": int(self._agent.config.proposal_every),
                "source_release_digest": self.source_release_digest,
            }
        )


__all__ = [
    "ORIGINAL_MAIN_COMMIT",
    "ORIGINAL_MAIN_FILES",
    "OriginalMainSourceError",
    "OriginalMainSourceReleaseV1",
    "PinnedOriginalMainAdapterV1",
]
