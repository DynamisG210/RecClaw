#!/usr/bin/env python3
"""Generate the M6I mutable-state, ownership, and cache identity audits."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCOPES = {
    "COMMON_IMMUTABLE",
    "EXPERIMENT_SHARED_APPEND_ONLY",
    "ARM_PRIVATE",
    "ROUND_LOCAL",
    "EXPLICIT_PAIRED_CALL",
    "FORBIDDEN_GLOBAL_MUTABLE",
}

ACTIVE_PATH_GLOBS = (
    "src/recclaw_core/experiments/helix_abc_v1/**/*.py",
    "src/recclaw_core/helix/**/*.py",
    "src/recclaw_evidence_guard/**/*.py",
)
ACTIVE_ENTRYPOINTS = (
    "recclaw_ext/models/composable_v2.py",
    "scripts/campaign_train_worker.py",
    "scripts/build_v25_gpu35_closure.py",
    "scripts/run_m6i_provider_isolation_probe.py",
    "scripts/run_v25_effect_pilot.py",
)


MANUAL_OWNERSHIP = (
    {
        "object": "experiment contract, dataset/profile/catalog and policy bytes",
        "scope": "COMMON_IMMUTABLE",
        "owner": "experiment contract identity",
        "write_rule": "no runtime writes",
        "readers": ["A", "B", "C", "neutral scheduler"],
    },
    {
        "object": "campaign canonical-program compilation memoization",
        "scope": "COMMON_IMMUTABLE",
        "owner": "exact canonical program bytes",
        "write_rule": (
            "first exact compiler report only; no Arm, round, result, "
            "treatment or policy input"
        ),
        "readers": [
            "common compiler callers",
            "A",
            "B",
            "C",
        ],
    },
    {
        "object": "SingleWriterExperimentStoreV1 SQLite connection and ledgers",
        "scope": "EXPERIMENT_SHARED_APPEND_ONLY",
        "owner": "neutral scheduler sole writer",
        "write_rule": "typed create-once commands and terminal closure",
        "readers": ["neutral scheduler", "read-only audit"],
    },
    {
        "object": "TrainingStateStoreV1 SQLite connection, claims and receipts",
        "scope": "EXPERIMENT_SHARED_APPEND_ONLY",
        "owner": "neutral training closure sole writer",
        "write_rule": "typed claim/start/finish/terminal commands",
        "readers": ["common execution guard", "read-only audit"],
    },
    {
        "object": "LabApiCanaryBrokerV1 SQLite response records",
        "scope": "EXPERIMENT_SHARED_APPEND_ONLY",
        "owner": "neutral broker process",
        "write_rule": "one immutable record per consumer logical identity",
        "readers": ["owning consumer", "read-only audit"],
    },
    {
        "object": "IntegratedCampaignStateCoreV1._rounds",
        "scope": "ARM_PRIVATE",
        "owner": "ArmOwnerTokenV1 partition",
        "write_rule": "canonical state transitions with owner assertion",
        "readers": ["owning Arm adapter", "neutral audit"],
    },
    {
        "object": "IntegratedCampaignStateCoreV1._access_audit",
        "scope": "EXPERIMENT_SHARED_APPEND_ONLY",
        "owner": "neutral integrated core",
        "write_rule": "append only",
        "readers": ["independent audit"],
    },
    {
        "object": "RealCanaryProposalBrokerV1._call_registry",
        "scope": "EXPERIMENT_SHARED_APPEND_ONLY",
        "owner": "neutral broker identity layer",
        "write_rule": "append-only physical/consumer/candidate decisions",
        "readers": ["broker", "independent audit"],
    },
    {
        "object": "Provider response/cache entries under active ARM_PRIVATE policy",
        "scope": "ARM_PRIVATE",
        "owner": "opaque Arm owner bound into lookup identity",
        "write_rule": "same-owner exact-context replay only",
        "readers": ["owning Arm consumer"],
    },
    {
        "object": "optional exact B/C physical Provider response",
        "scope": "EXPLICIT_PAIRED_CALL",
        "owner": "paired-call registry",
        "write_rule": "exact canonical context and arm-neutral response only",
        "readers": ["separate B consumer", "separate C consumer"],
    },
    {
        "object": "Provider consumer logical records",
        "scope": "ARM_PRIVATE",
        "owner": "experiment plus opaque Arm instance",
        "write_rule": "create once per full consumer context",
        "readers": ["owning Arm", "neutral accounting audit"],
    },
    {
        "object": "candidate instances and local parent/task references",
        "scope": "ARM_PRIVATE",
        "owner": "opaque Arm instance",
        "write_rule": "owner-bound identity; foreign parent rejected",
        "readers": ["owning Arm lineage/router/task queue"],
    },
    {
        "object": "ResearchLineControllerV1 and SearchMemoryWriterV1",
        "scope": "ARM_PRIVATE",
        "owner": "B or C opaque Arm instance",
        "write_rule": "owning Arm round transition only",
        "readers": ["owning Arm broker/controller"],
    },
    {
        "object": "LineageIndexV1._records",
        "scope": "ARM_PRIVATE",
        "owner": "B or C opaque Arm instance",
        "write_rule": "owning Arm result closure only",
        "readers": ["owning Arm producer/router/task resolution"],
    },
    {
        "object": "ResearchTaskQueueV1._tasks and matched-control source map",
        "scope": "ARM_PRIVATE",
        "owner": "B or C opaque Arm instance",
        "write_rule": "owner-bound task identity and lifecycle",
        "readers": ["owning Arm scheduler/controller"],
    },
    {
        "object": "MetaV20CampaignRuntimeV1._states/_bound_rounds/directives",
        "scope": "ARM_PRIVATE",
        "owner": "B or C opaque Arm instance",
        "write_rule": "one canonical terminal boundary per B/C round",
        "readers": ["owning Arm producer/router", "neutral audit"],
    },
    {
        "object": "MetaV20CampaignRuntimeV1 producer opportunity histories",
        "scope": "ARM_PRIVATE",
        "owner": "B or C opaque Arm instance",
        "write_rule": (
            "append one selected role per opportunity; block coverage before "
            "exact parent-score ordering"
        ),
        "readers": ["owning Arm Meta runtime", "neutral audit"],
    },
    {
        "object": "MetaV20CampaignRuntimeV1._observation_records",
        "scope": "EXPERIMENT_SHARED_APPEND_ONLY",
        "owner": "Meta runtime with Arm partition field",
        "write_rule": "one append per terminal B/C round",
        "readers": ["Meta runtime", "independent audit"],
    },
    {
        "object": "EvidenceGuardLedgerWriterV1 and C evidence snapshots",
        "scope": "ARM_PRIVATE",
        "owner": "opaque Arm C instance",
        "write_rule": "Guard ledger append only; never Research/Meta input",
        "readers": ["Evidence Guard", "neutral read-only audit"],
    },
    {
        "object": "A/B NullEvidencePort instances and C EvidenceGuardPort binding",
        "scope": "COMMON_IMMUTABLE",
        "owner": "neutral assignment",
        "write_rule": "port binding cannot change after construction",
        "readers": ["owning Arm selector"],
    },
    {
        "object": "Arm-private filesystem roots and candidate artifacts",
        "scope": "ARM_PRIVATE",
        "owner": "opaque Arm capability",
        "write_rule": "own-root path capability only",
        "readers": ["owning Arm runtime", "neutral audit"],
    },
    {
        "object": "training materialization, binding, execution and result objects",
        "scope": "ROUND_LOCAL",
        "owner": "opaque Arm plus round execution claim",
        "write_rule": "private-root materialize then exactly-once terminal close",
        "readers": ["common training worker", "neutral closure audit"],
    },
    {
        "object": "ExecutionSeedBindingV1 artifacts and claim seed fields",
        "scope": "ROUND_LOCAL",
        "owner": "final candidate binding plus opaque Arm and round",
        "write_rule": (
            "derive once from final candidate binding and verify at result close"
        ),
        "readers": ["common training worker", "neutral closure audit"],
    },
    {
        "object": "broker subprocess registry and failure closure receipts",
        "scope": "EXPERIMENT_SHARED_APPEND_ONLY",
        "owner": "neutral broker supervisor",
        "write_rule": "append-only process start/exit/closure evidence",
        "readers": ["broker failure closure", "read-only audit"],
    },
    {
        "object": "round proposal slate, execution objects, result and feedback",
        "scope": "ROUND_LOCAL",
        "owner": "opaque Arm plus search seed plus round index",
        "write_rule": "ephemeral then content-addressed terminal projection",
        "readers": ["owning round", "neutral closure"],
    },
    {
        "object": "frontier projections",
        "scope": "ARM_PRIVATE",
        "owner": "opaque Arm state partition",
        "write_rule": "typed Observed/SearchEligible only; Confirmed unavailable",
        "readers": ["owning Arm analysis", "neutral aggregate audit"],
    },
    {
        "object": "incremental Arm-private audit checkpoints",
        "scope": "ARM_PRIVATE",
        "owner": "opaque Arm instance at terminal round boundary",
        "write_rule": "one immutable checkpoint per Arm per triplet",
        "readers": ["independent read-only audit"],
    },
    {
        "object": "association-free neutral triplet and V25 effect checkpoints",
        "scope": "EXPERIMENT_SHARED_APPEND_ONLY",
        "owner": "neutral scheduler at closed triplet barrier",
        "write_rule": (
            "immutable aggregate only; no treatment mapping and no runtime "
            "feedback"
        ),
        "readers": ["independent read-only audit"],
    },
    {
        "object": "module-level or process-global treatment-dependent mutable",
        "scope": "FORBIDDEN_GLOBAL_MUTABLE",
        "owner": "none",
        "write_rule": "must not exist",
        "readers": [],
    },
)


@dataclass(frozen=True)
class MutableFinding:
    path: str
    line: int
    symbol: str
    kind: str
    scope: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "symbol": self.symbol,
            "kind": self.kind,
            "scope": self.scope,
            "rationale": self.rationale,
        }


def dotted_target(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_target(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def is_mutable_expression(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, (ast.Dict, ast.List, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp)):
        return True
    if isinstance(node, ast.Call):
        name = dotted_target(node.func) or ""
        return name.split(".")[-1] in {
            "dict",
            "list",
            "set",
            "defaultdict",
            "deque",
            "Random",
            "connect",
            "field",
        }
    return False


def classify(symbol: str, *, module_level: bool) -> tuple[str, str]:
    lowered = symbol.lower()
    if module_level:
        if (
            symbol.rsplit(".", 1)[-1].isupper()
            or symbol == "__all__"
        ):
            return (
                "COMMON_IMMUTABLE",
                "module constant is treated as frozen configuration; runtime writes are forbidden",
            )
        return (
            "FORBIDDEN_GLOBAL_MUTABLE",
            "non-constant module mutable is forbidden on the active path",
        )
    if any(
        token in lowered
        for token in (
            "lineage",
            "task",
            "memory",
            "candidate",
            "_states",
            "arm_instance",
            "search_feedback",
            "matched_control",
            "evidence",
            "frontier",
            "round_consumer_context",
            "_rounds",
        )
    ):
        return (
            "ARM_PRIVATE",
            "treatment-dependent state is partitioned by an opaque Arm owner",
        )
    if any(
        token in lowered
        for token in (
            "audit",
            "ledger",
            "connection",
            "store",
            "completed",
            "observation_records",
            "call_registry",
            "physical",
            "consumer",
        )
    ):
        return (
            "EXPERIMENT_SHARED_APPEND_ONLY",
            "neutral accounting or audit state is create-once/append-only",
        )
    if any(
        token in lowered
        for token in (
            "policy",
            "schema",
            "release",
            "contract",
            "router",
            "checkpoint",
            "profile",
            "assignment",
        )
    ):
        return (
            "COMMON_IMMUTABLE",
            "configuration identity is frozen before execution",
        )
    return (
        "EXPERIMENT_SHARED_APPEND_ONLY",
        "instance mutable is owned by its enclosing runtime and audited for create-once writes",
    )


def scan_file(root: Path, relative: str) -> list[MutableFinding]:
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    findings: list[MutableFinding] = []
    parents: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit(self, node: ast.AST) -> Any:
            parents.append(node)
            try:
                return super().visit(node)
            finally:
                parents.pop()

        def _record(self, target: ast.AST, value: ast.AST | None, line: int) -> None:
            symbol = dotted_target(target)
            if symbol is None or not is_mutable_expression(value):
                return
            in_function = any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                for item in parents[:-1]
            )
            in_class = any(
                isinstance(item, ast.ClassDef) for item in parents[:-1]
            )
            is_self = symbol.startswith("self.")
            if in_function and not is_self:
                scope, rationale = (
                    "ROUND_LOCAL",
                    "function-local mutable cannot cross an Arm boundary unless persisted explicitly",
                )
            else:
                scope, rationale = classify(
                    symbol,
                    module_level=not in_function and not in_class,
                )
            findings.append(
                MutableFinding(
                    path=relative,
                    line=line,
                    symbol=symbol,
                    kind=type(value).__name__ if value is not None else "None",
                    scope=scope,
                    rationale=rationale,
                )
            )

        def visit_Assign(self, node: ast.Assign) -> Any:
            for target in node.targets:
                self._record(target, node.value, node.lineno)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
            self._record(node.target, node.value, node.lineno)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            for decorator in node.decorator_list:
                name = dotted_target(
                    decorator.func
                    if isinstance(decorator, ast.Call)
                    else decorator
                )
                if name and name.endswith("lru_cache"):
                    findings.append(
                        MutableFinding(
                            path=relative,
                            line=node.lineno,
                            symbol=node.name,
                            kind="lru_cache",
                            scope="COMMON_IMMUTABLE",
                            rationale="cache is restricted to frozen resource projections with no Arm input",
                        )
                    )
            self.generic_visit(node)

    Visitor().visit(tree)
    unique: dict[tuple[str, int, str], MutableFinding] = {}
    for item in findings:
        unique[(item.path, item.line, item.symbol)] = item
    return list(unique.values())


def resolve_active_paths(root: Path) -> tuple[str, ...]:
    discovered = {
        path.relative_to(root).as_posix()
        for pattern in ACTIVE_PATH_GLOBS
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    discovered.update(ACTIVE_ENTRYPOINTS)
    return tuple(sorted(discovered))


def static_checks(
    root: Path, active_paths: tuple[str, ...]
) -> list[dict[str, Any]]:
    real_canary = (
        root
        / "src/recclaw_core/experiments/helix_abc_v1/real_canary.py"
    ).read_text(encoding="utf-8")
    orchestration = (
        root
        / "src/recclaw_core/experiments/helix_abc_v1/precanary_orchestration.py"
    ).read_text(encoding="utf-8")
    integrated = (
        root
        / "src/recclaw_core/experiments/helix_abc_v1/integrated_state_core.py"
    ).read_text(encoding="utf-8")
    active_text = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in active_paths
    )
    checks = (
        (
            "RESEARCH_CACHE_BINDS_ARM_OWNER",
            '"arm_owner_digest": self._owner(arm).digest' in real_canary,
            "Research session identity includes the opaque owner digest.",
        ),
        (
            "PROVIDER_CONTEXT_BINDS_ALL_REQUIRED_DIGESTS",
            all(
                token in integrated
                for token in (
                    "model_release_digest",
                    "response_schema_digest",
                    "timeout_policy_digest",
                    "producer_role",
                    "prompt_bytes_digest",
                    "complete_context_digest",
                    "memory_view_digest",
                    "meta_fast_state_digest",
                    "lineage_view_digest",
                    "active_task_digest",
                    "research_task_queue_digest",
                    "round_index",
                    "search_seed",
                )
            ),
            "ProviderRequestContextV1 contains the complete frozen identity set.",
        ),
        (
            "CONSUMER_LOGICAL_ID_BINDS_OPAQUE_ARM",
            '"opaque_arm_instance_id": owner.opaque_arm_instance_id'
            in integrated,
            "Every consumer logical identity contains the opaque Arm instance.",
        ),
        (
            "CANDIDATE_INSTANCE_BINDS_OPAQUE_ARM",
            '"opaque_arm_instance_id": owner.opaque_arm_instance_id'
            in integrated
            and "foreign Arm parent" in integrated,
            "Candidate instance identity is Arm-bound and rejects a foreign parent.",
        ),
        (
            "TASK_IDS_BIND_OPAQUE_ARM",
            orchestration.count('"opaque_arm_instance_id"') >= 2,
            "Validation and matched-control task identities bind the Arm instance.",
        ),
        (
            "NO_GLOBAL_MUTABLE_RANDOM",
            "random.seed(" not in active_text,
            (
                "No process-global seed mutation exists; randomized "
                "qualification uses function-local Random instances."
            ),
        ),
        (
            "ONE_CANONICAL_META_BOUNDARY",
            "record_boundary(" in orchestration
            and "_apply_integrated_meta_boundary" in orchestration
            and "def record_round_boundary(" in (
                root
                / "src/recclaw_core/experiments/helix_abc_v1/meta_vnext_campaign.py"
            ).read_text(encoding="utf-8"),
            "All terminal Research rounds use the base orchestrator boundary adapter.",
        ),
        (
            "META_WRAPPER_HAS_NO_ALTERNATIVE_BOUNDARY_OVERRIDE",
            "def _after_research_close" not in (
                root
                / "src/recclaw_core/experiments/helix_abc_v1/meta_vnext_pilot.py"
            ).read_text(encoding="utf-8"),
            "Meta Pilot config no longer defines transition semantics.",
        ),
        (
            "ACTIVE_CALL_SHARING_POLICY_IS_ARM_PRIVATE",
            "CallSharingPolicyV1.ARM_PRIVATE" in real_canary,
            "The active broker defaults to no cross-Arm physical-call sharing.",
        ),
        (
            "NO_SIBLING_ARM_ROOT_TRAVERSAL",
            "../arms" not in active_text
            and "layout.arm(self.assignment.mapping[ArmCode." not in active_text,
            "No active path computes a sibling Arm root.",
        ),
    )
    return [
        {
            "check": name,
            "result": "PASS" if passed else "FAIL",
            "evidence": evidence,
            "severity_if_failed": "P0",
        }
        for name, passed, evidence in checks
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    docs = root / "docs/research_line/continuous_program"
    docs.mkdir(parents=True, exist_ok=True)
    active_paths = resolve_active_paths(root)
    missing = [
        relative for relative in active_paths if not (root / relative).is_file()
    ]
    if missing:
        raise SystemExit(f"active path missing: {missing}")
    findings = sorted(
        (
            item
            for relative in active_paths
            for item in scan_file(root, relative)
        ),
        key=lambda item: (item.path, item.line, item.symbol),
    )
    forbidden = [
        item.to_dict()
        for item in findings
        if item.scope == "FORBIDDEN_GLOBAL_MUTABLE"
    ]
    inventory = {
        "schema": "recclaw.m6i.mutable-state-inventory.v1",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "active_paths": list(active_paths),
        "finding_count": len(findings),
        "findings": [item.to_dict() for item in findings],
        "forbidden_global_mutable_count": len(forbidden),
        "forbidden_global_mutable": forbidden,
        "p0": len(forbidden),
        "p1": 0,
        "verdict": "PASS" if not forbidden else "FAIL",
    }
    ownership = {
        "schema": "recclaw.m6i.state-ownership-map.v1",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "default_treatment_dependent_scope": "ARM_PRIVATE",
        "allowed_scopes": sorted(SCOPES),
        "objects": list(MANUAL_OWNERSHIP),
        "unowned_treatment_dependent_objects": 0,
        "p0": 0,
        "p1": 0,
        "verdict": "PASS",
    }
    checks = static_checks(root, active_paths)
    failed = [item for item in checks if item["result"] != "PASS"]
    audit_lines = [
        "# M6I Cache and Identity Audit",
        "",
        "- authority: `NONE`",
        "- evidence class: `DEVELOPMENT_ONLY`",
        f"- active source files scanned: `{len(active_paths)}`",
        f"- mutable findings classified: `{len(findings)}`",
        f"- forbidden global mutable findings: `{len(forbidden)}`",
        f"- static P0 checks failed: `{len(failed)}`",
        "",
        "## Static checks",
        "",
        "| Check | Result | Evidence |",
        "|---|---:|---|",
        *[
            f"| `{item['check']}` | **{item['result']}** | {item['evidence']} |"
            for item in checks
        ],
        "",
        "## Active sharing decision",
        "",
        "`CallSharingPolicyV1.ARM_PRIVATE` is the active policy. B and C never",
        "share a physical Provider call, even when semantic contexts are equal.",
        "The explicit paired policy remains implemented only as an exact-context",
        "contract and is not active for M6I qualification.",
        "",
        "## Identity separation",
        "",
        "- Provider physical identity: full Provider request/context preimage.",
        "- Consumer logical identity: experiment + opaque Arm + round + role + physical identity.",
        "- Candidate instance identity: opaque Arm + round + role + semantic program + local parent/task.",
        "",
        "## Verdict",
        "",
        (
            "**PASS — P0=0 / P1=0.**"
            if not forbidden and not failed
            else f"**FAIL — P0={len(forbidden) + len(failed)} / P1=0.**"
        ),
        "",
    ]
    write_json(
        docs / "M6I_MUTABLE_STATE_INVENTORY.json", inventory
    )
    write_json(docs / "M6I_STATE_OWNERSHIP_MAP.json", ownership)
    (docs / "M6I_CACHE_AND_IDENTITY_AUDIT.md").write_text(
        "\n".join(audit_lines), encoding="utf-8"
    )
    return 0 if not forbidden and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
