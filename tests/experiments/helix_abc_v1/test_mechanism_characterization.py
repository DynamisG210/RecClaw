from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/mechanism_characterization.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "q2_mechanism_characterization", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
classify_mechanism_evidence = MODULE.classify_mechanism_evidence
full_ablation_allowed = MODULE.full_ablation_allowed
load_contract = MODULE.load_contract
q3_evidence_package = MODULE.q3_evidence_package
validate_selected_package = MODULE.validate_selected_package
evaluate_probe_statistics = MODULE.evaluate_probe_statistics
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research_line/vnext/Q2_MECHANISM_PROBE_CONTRACT.json"
)
PACKAGE_PATH = (
    REPO_ROOT
    / "results/research_line/q1_prompt_contract_20260802_01/selected/enriched"
    / "candidate_package.json"
)
CANDIDATE_PATH = (
    REPO_ROOT
    / "results/research_line/q1_prompt_contract_20260802_01/selected/enriched"
    / "candidates/diagnosis/innovation-candidate-a556e754f5a3fc35ef7eef04"
    / "recclaw_ext/candidate.py"
)
IDEA_QUALITY_PATH = (
    REPO_ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/idea_quality.py"
)
Q2_RESULT_ROOT = (
    REPO_ROOT
    / "results/research_line/q2_mechanism_characterization_20260802_01"
)
Q2_CANONICAL_WRAPPER_PATH = (
    REPO_ROOT
    / "docs/research_line/vnext/"
    "Q2_MECHANISM_CHARACTERIZATION_CANONICAL_RECEIPT.json"
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_binds_the_unique_q1_package_and_development_boundary() -> None:
    contract = _json(CONTRACT_PATH)
    package = _json(PACKAGE_PATH)

    assert contract["status"] == "FROZEN_BEFORE_Q2_OUTCOME"
    assert contract["development_only"] is True
    assert contract["scientific_effect_claim_allowed"] is False
    assert contract["input"] == {
        "q1_commit": "d10083908a64a0d05a0cbbb1a6eddc2055b2f710",
        "q1_tree": "4f44fa0165292a442ec9e5b3ddafb2335e3ae36c",
        "q1_physical_receipt_path": (
            "results/research_line/q1_prompt_contract_20260802_01/"
            "Q1_CANONICAL_RECEIPT.json"
        ),
        "q1_physical_receipt_sha256": (
            "cbb2e8db9ee07cdb923061686dc8f7a44e8eddb65c504ab7803b079ee8935005"
        ),
        "candidate_package_path": (
            "results/research_line/q1_prompt_contract_20260802_01/selected/"
            "enriched/candidate_package.json"
        ),
        "candidate_package_file_sha256": (
            "f828081be48a0bfc12b219c24e9e82afdc0905154805f7a9e27a31844341e462"
        ),
        "candidate_package_digest": (
            "0f4567d20b96d84b2cddcc39de6f1bea5be531966a8a210ccd95bb97f6010783"
        ),
        "candidate_source_tree_digest": (
            "ba87decc9e72d8fde853ed17de1520cd9de9a2b436837ec2868734af2ff7c757"
        ),
        "candidate_source_sha256": (
            "30b0d22b0d25ec7444d5dd3137ffce4bdc10bc611099ff48aac5e711b1e180bd"
        ),
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "research_spec_digest": (
            "1871d8c0dfd6f36cbce380a78e1e1a11a610629d8770522d867fc2f13d074988"
        ),
        "held_out_reads_before_freeze": 0,
    }
    assert _sha256(PACKAGE_PATH) == contract["input"][
        "candidate_package_file_sha256"
    ]
    assert package["package"]["source_tree_digest"] == contract["input"][
        "candidate_source_tree_digest"
    ]
    assert package["package"]["research_spec_digest"] == contract["input"][
        "research_spec_digest"
    ]
    assert _sha256(CANDIDATE_PATH) == contract["input"]["candidate_source_sha256"]


def test_contract_freezes_origin_blind_probe_and_resource_rules() -> None:
    contract = _json(CONTRACT_PATH)

    assert contract["probe_population"]["held_out_reads"] == 0
    assert contract["probe_population"]["seed"] == 54102
    assert contract["probe_population"]["probe_train_batch_indices"] == [0, 15, 31]
    assert contract["probe_population"]["maximum_examples_per_probe_batch"] == 32
    assert contract["resource_probe"] == {
        "must_run_before_mechanism_effect_interpretation": True,
        "execution_purpose": "RESOURCE_PROBE_ONLY",
        "prefix_contract_sha256": (
            "c87190d7a1a0b997f2f513c8bc7605a5e9c7d2cf6ec3ad6cf3c6f2c7351e446c"
        ),
        "epochs": 3,
        "deadline_seconds": 300,
        "seed": 54102,
        "train_batch_indices": "0..31 inclusive",
        "eval_batch_indices": "0..63 inclusive",
        "telemetry": [
            "train and eval peak allocated MiB",
            "train and eval peak reserved MiB",
            "batch throughput",
            "loss trend",
            "phase wall time",
        ],
        "resource_failure_updates_mechanism_effect": False,
        "missingness_states": ["RESOURCE_CENSORED", "RESOURCE_DEFERRED"],
    }
    assert contract["cheap_probes"]["discriminative_prediction"]["comparison"].startswith(
        "learned top-one removal versus deterministic"
    )
    assert contract["authority_boundaries"]["candidate_specific_source_patch_allowed"] is False
    assert contract["authority_boundaries"][
        "protocol_or_outcome_rule_change_after_freeze_allowed"
    ] is False


def test_state_machine_keeps_resource_missingness_out_of_mechanism_effect() -> None:
    contract = _json(CONTRACT_PATH)
    state_machine = contract["state_machine"]

    assert state_machine["states"] == [
        "NOT_ASSESSED",
        "INACTIVE",
        "ACTIVE_SUPPORTED",
        "ACTIVE_CONTRADICTED",
        "NON_IDENTIFIABLE",
    ]
    assert state_machine["resource_missingness_is_orthogonal"] is True
    assert state_machine["resource_missingness_may_change_state"] is False
    assert state_machine["mechanism_effect_update_allowed_only_for"] == [
        "ACTIVE_SUPPORTED",
        "ACTIVE_CONTRADICTED",
    ]
    assert contract["resource_probe"]["resource_failure_updates_mechanism_effect"] is False


def test_selected_candidate_static_path_is_recorded_without_calling_it_effect() -> None:
    contract = _json(CONTRACT_PATH)
    source = CANDIDATE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    candidate = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"
    )
    method_names = {
        node.name for node in candidate.body if isinstance(node, ast.FunctionDef)
    }
    assigned_attributes = {
        node.targets[0].attr
        for node in ast.walk(candidate)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
    }

    assert [base.id for base in candidate.bases if isinstance(base, ast.Name)] == ["BPR"]
    assert {
        "set_mechanism_enabled",
        "_history_summary",
        "_stability_penalty",
        "calculate_loss",
        "predict",
        "full_sort_predict",
    } <= method_names
    assert {"_mechanism_enabled", "support_gate"} <= assigned_attributes
    assert contract["static_input_facts_not_counted_as_probe_results"][
        "candidate_base_class"
    ] == "recbole.model.general_recommender.bpr:BPR"
    assert contract["static_input_facts_not_counted_as_probe_results"][
        "candidate_active_path"
    ].startswith("user embedding plus softmax-weighted training-history")
    assert contract["authority_boundaries"]["probe_is_scientific_effect"] is False


def test_q2_declared_parent_check_is_stricter_than_q1_bpr_reference() -> None:
    contract = _json(CONTRACT_PATH)
    q1_source = IDEA_QUALITY_PATH.read_text(encoding="utf-8")

    assert "from recbole.model.general_recommender.bpr import BPR" in q1_source
    assert "baseline = BPR(config, dataset)" in q1_source
    assert contract["declared_mechanism"]["closest_parent_mechanism_id"] == (
        "LIGHTGCN__LGCN_DUAL_PATH"
    )
    assert contract["declared_mechanism"]["closest_parent_entrypoint"] == (
        "recclaw_ext.models.composable_v2:LightGCNComposableV2"
    )
    assert contract["cheap_probes"]["mechanism_off_parent_equivalence"][
        "failure_disposition"
    ] == "NON_IDENTIFIABLE"


def _passing_probe_evidence() -> dict[str, dict[str, str]]:
    return {
        "loss_participation": {"status": "PASS"},
        "gate_activation": {"status": "PASS"},
        "routing_and_propagation": {"status": "PASS"},
        "target_conditioning": {"status": "PASS"},
        "mechanism_off_parent_equivalence": {"status": "PASS"},
        "discriminative_prediction": {"status": "SUPPORTED"},
    }


def test_consumer_validates_the_physical_selected_candidate_root() -> None:
    contract = load_contract(CONTRACT_PATH)
    selected = validate_selected_package(REPO_ROOT, contract)

    assert selected["candidate_package_path"] == PACKAGE_PATH
    assert selected["candidate_source_path"] == CANDIDATE_PATH
    assert selected["candidate_root"].name == (
        "innovation-candidate-a556e754f5a3fc35ef7eef04"
    )


def test_state_precedence_distinguishes_inactive_and_non_identifiable() -> None:
    evidence = _passing_probe_evidence()
    assert classify_mechanism_evidence(evidence) == "ACTIVE_SUPPORTED"

    contradicted = {**evidence, "discriminative_prediction": {"status": "CONTRADICTED"}}
    assert classify_mechanism_evidence(contradicted) == "ACTIVE_CONTRADICTED"

    inactive = {**evidence, "gate_activation": {"status": "FAIL"}}
    assert classify_mechanism_evidence(inactive) == "INACTIVE"

    non_identifiable = {
        **evidence,
        "mechanism_off_parent_equivalence": {"status": "FAIL"},
    }
    assert classify_mechanism_evidence(non_identifiable) == "NON_IDENTIFIABLE"

    not_assessed = {**evidence, "loss_participation": {"status": "NOT_RUN"}}
    assert classify_mechanism_evidence(not_assessed) == "NOT_ASSESSED"


def test_failed_identifiability_is_decisive_without_a_checkpoint() -> None:
    evidence = {
        "loss_participation": {"status": "PASS"},
        "gate_activation": {"status": "NOT_ASSESSED_PROTOCOL_NO_CHECKPOINT"},
        "routing_and_propagation": {"status": "PASS"},
        "target_conditioning": {"status": "FAIL"},
        "mechanism_off_parent_equivalence": {"status": "FAIL"},
        "discriminative_prediction": {
            "status": "NOT_ASSESSED_PROTOCOL_NO_CHECKPOINT"
        },
    }

    assert classify_mechanism_evidence(evidence) == "NON_IDENTIFIABLE"


def test_resource_missingness_is_orthogonal_and_blocks_full_ablation() -> None:
    evidence = _passing_probe_evidence()
    state = classify_mechanism_evidence(evidence)

    assert full_ablation_allowed(
        state=state,
        evidence=evidence,
        resource_status="SUCCESS",
    )
    assert not full_ablation_allowed(
        state=state,
        evidence=evidence,
        resource_status="RESOURCE_DEFERRED",
    )
    q3 = q3_evidence_package(
        contract_sha256="a" * 64,
        physical_result_sha256="b" * 64,
        state=state,
        evidence=evidence,
        resource_status="RESOURCE_DEFERRED",
        full_ablation_executed=False,
    )
    assert q3["mechanism_state"] == "ACTIVE_SUPPORTED"
    assert q3["mechanism_effect_update_allowed"] is True
    assert q3["resource_updates_mechanism_effect"] is False
    assert q3["full_ablation_eligible"] is False


def test_physical_statistics_use_frozen_thresholds_and_surface_parent_mismatch() -> None:
    contract = load_contract(CONTRACT_PATH)
    statistics = {
        "loss_participation": {
            "full_loss": 0.8,
            "mechanism_off_loss": 0.7,
            "stability_contribution": 0.01,
            "full_off_abs_delta": 0.1,
            "mechanism_gradient_l2": {"support_gate.weight": 0.02},
        },
        "gate_activation": {
            "distinct_user_count": 40,
            "eligible_user_count": 20,
            "raw_logit_std": 0.2,
            "support_saturation_rate": 0.0,
            "distinct_centered_logit_signatures": 10,
        },
        "routing_and_propagation": {
            "support_gate_call_count": 3,
            "positive_score_max_abs_delta": 0.3,
            "pairwise_margin_max_abs_delta": 0.2,
            "candidate_declared_graph_propagation_present": False,
        },
        "target_conditioning": {
            "compared_user_count": 8,
            "max_centered_common_logit_delta": 0.0,
            "pairwise_order_change_count": 0,
        },
        "mechanism_off_parent_equivalence": {
            "max_abs_deltas": {
                "predict": 0.2,
                "full_sort_predict": 0.3,
                "calculate_loss": 0.1,
                "shared_parameter_gradients": 0.4,
            },
            "reference_max_abs_values": {
                "predict": 1.0,
                "full_sort_predict": 1.0,
                "calculate_loss": 1.0,
                "shared_parameter_gradients": 1.0,
            },
        },
        "discriminative_prediction": {
            "eligible_example_count": 12,
            "learned_removal_mean_abs_margin_damage": 0.2,
            "control_removal_mean_abs_margin_damage": 0.1,
        },
    }

    evidence = evaluate_probe_statistics(statistics, contract)

    assert evidence["loss_participation"]["status"] == "PASS"
    assert evidence["gate_activation"]["status"] == "PASS"
    assert evidence["routing_and_propagation"]["status"] == "PASS"
    assert evidence["target_conditioning"]["status"] == "FAIL"
    assert evidence["mechanism_off_parent_equivalence"]["status"] == "FAIL"
    assert evidence["discriminative_prediction"]["status"] == "SUPPORTED"
    assert classify_mechanism_evidence(evidence) == "NON_IDENTIFIABLE"


def test_resource_telemetry_restore_restores_the_original_loss_callable(
    tmp_path: Path,
) -> None:
    worker_path = REPO_ROOT / "scripts/campaign_train_worker.py"
    worker_spec = importlib.util.spec_from_file_location(
        "q2_campaign_train_worker", worker_path
    )
    assert worker_spec is not None and worker_spec.loader is not None
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)

    class TrainLoader:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            yield "train"

    class ValidLoader:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            yield "valid"

    class Model:
        def parameters(self):
            return ()

        def calculate_loss(self, interaction: object) -> object:
            return interaction

    class Trainer:
        def __init__(self) -> None:
            self.model = Model()

        def _train_epoch(self, *args: object, **kwargs: object) -> float:
            return 0.0

        def _valid_epoch(self, *args: object, **kwargs: object):
            return 0.0, {}

    trainer = Trainer()
    original_loss = trainer.model.calculate_loss
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    _telemetry, restore = worker._install_resource_telemetry(
        trainer,
        torch=fake_torch,
        train_data=TrainLoader(),
        valid_data=ValidLoader(),
        telemetry_path=tmp_path / "telemetry.json",
        prefix_contract={"train_batch_indices": [0], "eval_batch_indices": [0]},
        preallocated_train_batches=("train",),
        preallocated_valid_batches=("valid",),
        worker_started_ns=0,
    )
    assert trainer.model.calculate_loss("wrapped") == "wrapped"
    assert trainer.model.calculate_loss is not original_loss

    restore()

    assert trainer.model.calculate_loss == original_loss
    assert trainer.model.calculate_loss("restored") == "restored"


def test_canonical_receipt_binds_negative_evidence_and_telemetry_disclosure() -> None:
    wrapper = _json(Q2_CANONICAL_WRAPPER_PATH)
    receipt_path = Q2_RESULT_ROOT / "Q2_CANONICAL_RECEIPT.json"
    receipt = _json(receipt_path)
    physical_path = Q2_RESULT_ROOT / "Q2_PHYSICAL_RESULT.json"
    q3_path = Q2_RESULT_ROOT / "Q3_MECHANISM_EVIDENCE_PACKAGE.json"

    assert wrapper["physical_receipt_sha256"] == _sha256(receipt_path)
    assert wrapper["physical_result_sha256"] == _sha256(physical_path)
    assert wrapper["q3_input_sha256"] == _sha256(q3_path)
    assert receipt["decision"]["mechanism_state"] == "NON_IDENTIFIABLE"
    assert receipt["decision"]["mechanism_effect_update_allowed"] is False
    assert receipt["decision"]["full_ablation_executed"] is False
    assert receipt["resource_probe"]["status"] == "RESOURCE_CENSORED"
    assert receipt["protocol_consumer_missingness"]["is_resource_censoring"] is False
    assert receipt["telemetry_integrity"]["pollution_occurred"] is True
    assert receipt["telemetry_integrity"][
        "post_resource_field_belongs_to_resource_result"
    ] is False
    assert _sha256(
        Q2_RESULT_ROOT / "RESOURCE_TELEMETRY_CONTAMINATED_POST_RESOURCE.json"
    ) == receipt["telemetry_integrity"]["post_seal_polluted_telemetry_sha256"]
    assert _sha256(
        Q2_RESULT_ROOT / "RESOURCE_TELEMETRY_AUTHORITATIVE.json"
    ) == receipt["telemetry_integrity"]["resource_result_bound_telemetry_sha256"]
