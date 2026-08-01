from __future__ import annotations

from pathlib import Path
import json

import jsonschema
import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
    AVAILABLE_DEPENDENCIES,
    IMPLEMENTATION_TOKEN_CEILING,
    PROPOSAL_TOKEN_CEILING,
    PROTOCOL_REQUIREMENTS,
    RECBole_ROOT,
    SEARCH_DATASET_ROOT,
    _materialize_and_qualify,
    call_contract_for_side,
    derive_fresh_r1_proposal_schema,
    evaluate_gate,
    render_implementation_prompt,
    render_proposal_prompt,
    retry_eligible,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    validate_provider_strict_schema,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_bindings,
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.training_filesystem import (
    build_training_filesystem_capability,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    QualificationStageV1,
    QualificationStatusV1,
)


ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"


def _strict_proposal_schema() -> dict[str, object]:
    base = json.loads(
        (
            RESOURCE_ROOT
            / "fresh_open_spec_proposal_response_v4_provider.schema.json"
        ).read_bytes()
    )
    delta = json.loads(
        (RESOURCE_ROOT / "fresh_r1_proposal_schema_delta_v1.json").read_bytes()
    )
    return derive_fresh_r1_proposal_schema(base, delta)


def _interface_spec() -> object:
    draft = {
        "producer_role": "mechanism_composer",
        "hypothesis": "A trainable score offset changes pairwise ranking behavior.",
        "mechanism_change": "Add a candidate-local trainable score interaction.",
        "competing_explanation": "The result comes only from inherited BPR.",
        "matched_control_requirement": "Compare against BPR with the same seed.",
        "implementation_requirements": [
            "RecBole GeneralRecommender entrypoint",
            "Override all scoring and loss methods",
        ],
        "expected_evidence": ["Scores and loss differ from matched BPR."],
        "falsifier": "Reject when scores and loss equal inherited BPR.",
        "compatibility_requirements": [
            "general collaborative filtering",
            "pairwise input",
        ],
        "high_change_justification": (
            "Adds a trainable interaction absent from the catalog."
        ),
        "current_profile_expressibility_claim": "NOT_EXPRESSIBLE",
        "resolution_facts": {
            "requested_current_semantics_digest": None,
            "capability_diff": ["trainable score interaction"],
            "high_change_dimensions": ["INTERACTION_STRUCTURE"],
            "required_dependencies": ["recbole-runtime", "torch"],
            "required_budget": {
                "implementation_token_ceiling": 20_000,
                "qualification_gpu_minutes": 10,
                "qualification_wall_minutes": 30,
            },
        },
    }
    spec, _facts = project_open_producer_draft(
        draft,
        bindings=frozen_search_bindings(
            context_ref="fresh-r1-r2-prefreeze-context-v1",
            context_digest=sha256_digest({"fixture": "corrected-r1"}),
        ),
    )
    return spec


def _implementation(source: str) -> dict[str, object]:
    return {
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "files": [
            {"path": "recclaw_ext/__init__.py", "content": "# fresh\n"},
            {"path": "recclaw_ext/candidate.py", "content": source},
        ],
        "implementation_summary": "Trainable score interaction fixture.",
    }


def _qualify_source(tmp_path: Path, *, label: str, source: str):
    return _materialize_and_qualify(
        repo_root=ROOT,
        side_root=tmp_path / label,
        slot_id="slot-01",
        seed=20260801,
        spec=_interface_spec(),
        implementation=_implementation(source),
        implementation_prompt_digest=bytes_sha256(
            (RESOURCE_ROOT / "fresh_r1_implementer_prompt_v1.txt").read_bytes()
        ),
        tool_policy_digest=bytes_sha256(
            (RESOURCE_ROOT / "fresh_open_spec_tool_policy_v1.json").read_bytes()
        ),
    )


def test_formal_implementation_schema_is_strict_and_candidate_local() -> None:
    schema = __import__("json").loads(
        (RESOURCE_ROOT / "fresh_r1_implementation_response_v1.schema.json").read_bytes()
    )
    jsonschema.validators.validator_for(schema).check_schema(schema)
    response = {
        "schema": "recclaw.research-line.fresh-r1-implementation-response.v1",
        "proposals": [
            {
                "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
                "files": [
                    {"path": "recclaw_ext/__init__.py", "content": "# fresh\n"},
                    {
                        "path": "recclaw_ext/candidate.py",
                        "content": "class FreshCandidateModel:\n    pass\n",
                    },
                ],
                "implementation_summary": "fixture",
            }
        ],
    }
    jsonschema.validate(response, schema)


def test_prompts_preserve_slot_role_and_blind_implementation_boundary() -> None:
    proposal_template = (
        RESOURCE_ROOT / "fresh_open_spec_proposal_prompt_v1.txt"
    ).read_text(encoding="utf-8")
    rendered = render_proposal_prompt(
        proposal_template,
        side_identity="recclaw-fresh-r1-side-a-v1",
        logical_slot_id="slot-01",
        proposal_seed=41001,
        producer_role="mechanism_composer",
    )
    assert "slot-01" in rendered
    assert "41001" in rendered
    assert "mechanism_composer" in rendered
    assert all(value in rendered for value in PROTOCOL_REQUIREMENTS)
    implementation_template = (
        RESOURCE_ROOT / "fresh_r1_implementer_prompt_v1.txt"
    ).read_text(encoding="utf-8")
    implementation = render_implementation_prompt(
        implementation_template,
        {
            "blind_candidate_id": "innovation-candidate-123",
            "blind_research_spec": {"hypothesis": "new relation"},
            "candidate_local_write_allowlist": ["recclaw_ext/candidate.py"],
            "schema": "recclaw.shared-implementer-request.v1",
            "service_policy": {"candidate_local_write_only": True},
        },
    )
    assert "innovation-candidate-123" in implementation
    assert "side_a" not in implementation
    assert "producer_role" not in implementation


def test_retry_policy_is_exactly_transient_only() -> None:
    assert retry_eligible({"http_status": 408})
    assert retry_eligible({"http_status": 429})
    assert retry_eligible({"http_status": 503})
    assert retry_eligible({"failure_class": "TIMEOUT"})
    assert retry_eligible({"exception_type": "ConnectionResetError"})
    assert not retry_eligible({"http_status": 400})
    assert not retry_eligible({"http_status": 401})
    assert not retry_eligible({"failure_class": "SCHEMA_VALIDATION_FAILURE"})
    assert not retry_eligible({"exception_type": "URLError"})


def test_gate_requires_both_sides_and_real_behavior_change() -> None:
    records = {}
    for side in ("side_a", "side_b"):
        records[side] = [
            {
                "producer_role": "mechanism_composer" if index % 2 == 0 else "frontier_architect",
                "qualification_status": "PASS" if index < 2 else None,
                "real_mechanism_change": index == 0,
                "spec_digest": f"{side}-{index}",
            }
            for index in range(4)
        ]
    assert evaluate_gate(records)["pass"] is True
    records["side_b"][1]["qualification_status"] = "FAIL"
    assert evaluate_gate(records)["pass"] is False


def test_corrected_provider_schema_rejects_paraphrase_and_accepts_exact_tokens() -> None:
    schema = _strict_proposal_schema()
    validate_provider_strict_schema(schema)
    response = json.loads(
        (
            RESOURCE_ROOT
            / "fresh_open_spec_v4_duplicate_arrays_negative_fixture.json"
        ).read_bytes()
    )
    proposal = response["proposals"][0]
    proposal["compatibility_requirements"] = list(PROTOCOL_REQUIREMENTS)
    proposal["resolution_facts"]["required_dependencies"] = list(
        AVAILABLE_DEPENDENCIES
    )
    jsonschema.validate(response, schema)
    paraphrased = json.loads(json.dumps(response))
    paraphrased["proposals"][0]["compatibility_requirements"] = [
        "Use the frozen dataset and split."
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(paraphrased, schema)


@pytest.mark.parametrize(
    ("label", "initialization", "loss_term"),
    [
        (
            "config_index",
            "self.width = int(config['embedding_size'])\n"
            "        self.fresh_scale = torch.nn.Parameter("
            "torch.tensor(1.0 / self.width))",
            "self.fresh_scale.square()",
        ),
        (
            "literal_default",
            "self.temperature = 0.125\n"
            "        self.fresh_scale = torch.nn.Parameter("
            "torch.tensor(self.temperature))",
            "self.fresh_scale.square()",
        ),
        (
            "initialized_regularizer",
            "self.reg_loss = EmbLoss()\n"
            "        self.fresh_scale = torch.nn.Parameter(torch.tensor(0.125))",
            "self.fresh_scale.square() + 1e-8 * self.reg_loss("
            "self.user_embedding(interaction[self.USER_ID]))",
        ),
    ],
)
def test_real_recbole_interface_contract_materializes_and_qualifies(
    tmp_path: Path,
    label: str,
    initialization: str,
    loss_term: str,
) -> None:
    source = (
        "import torch\n"
        "from recbole.model.general_recommender.bpr import BPR\n"
        "from recbole.model.loss import EmbLoss\n\n"
        "class FreshCandidateModel(BPR):\n"
        "    def __init__(self, config, dataset):\n"
        "        super().__init__(config, dataset)\n"
        f"        {initialization}\n\n"
        "    def calculate_loss(self, interaction):\n"
        f"        return super().calculate_loss(interaction) + {loss_term}\n\n"
        "    def predict(self, interaction):\n"
        "        return super().predict(interaction) + self.fresh_scale\n\n"
        "    def full_sort_predict(self, interaction):\n"
        "        return super().full_sort_predict(interaction) + self.fresh_scale\n"
    )
    materialized, qualification, behavior = _qualify_source(
        tmp_path,
        label=label,
        source=source,
    )
    candidate_root = (
        tmp_path
        / label
        / "candidates/slot-01"
        / str(materialized.shared_request["blind_candidate_id"])
    )
    assert candidate_root.is_dir()
    assert candidate_root.name == materialized.shared_request["blind_candidate_id"]
    assert qualification.receipt.status is QualificationStatusV1.PASS
    assert behavior["probe_status"] == "PASS"


@pytest.mark.parametrize(
    ("label", "source_fragment", "expected_stage", "exception_type"),
    [
        (
            "config_get",
            "self.fresh_scale = torch.nn.Parameter("
            "torch.tensor(float(config.get('scale', 0.125))))",
            QualificationStageV1.CONSTRUCTION,
            "ATTRIBUTEERROR",
        ),
        (
            "float_none",
            "self.fresh_scale = torch.nn.Parameter("
            "torch.tensor(float(config['candidate_scale'])))",
            QualificationStageV1.CONSTRUCTION,
            "TYPEERROR",
        ),
        (
            "uninitialized_attribute",
            "self.fresh_scale = torch.nn.Parameter(torch.tensor(0.125))",
            QualificationStageV1.API_CONTRACT,
            "ATTRIBUTEERROR",
        ),
    ],
)
def test_real_qualifier_reproduces_observed_recbole_interface_failures(
    tmp_path: Path,
    label: str,
    source_fragment: str,
    expected_stage: QualificationStageV1,
    exception_type: str,
) -> None:
    invalid_loss = (
        "self.reg_loss(self.user_embedding(interaction[self.USER_ID]))"
        if label == "uninitialized_attribute"
        else "self.fresh_scale.square()"
    )
    source = (
        "import torch\n"
        "from recbole.model.general_recommender.bpr import BPR\n\n"
        "class FreshCandidateModel(BPR):\n"
        "    def __init__(self, config, dataset):\n"
        "        super().__init__(config, dataset)\n"
        f"        {source_fragment}\n\n"
        "    def calculate_loss(self, interaction):\n"
        f"        return super().calculate_loss(interaction) + {invalid_loss}\n\n"
        "    def predict(self, interaction):\n"
        "        return super().predict(interaction) + self.fresh_scale\n\n"
        "    def full_sort_predict(self, interaction):\n"
        "        return super().full_sort_predict(interaction) + self.fresh_scale\n"
    )
    materialized, qualification, behavior = _qualify_source(
        tmp_path,
        label=label,
        source=source,
    )
    candidate_root = (
        tmp_path
        / label
        / "candidates/slot-01"
        / str(materialized.shared_request["blind_candidate_id"])
    )
    assert candidate_root.is_dir()
    assert qualification.receipt.status is QualificationStatusV1.FAIL
    assert qualification.receipt.stage is expected_stage
    assert qualification.failure_detail["reason_code"] == exception_type
    assert behavior == {}


def test_proposal_and_implementer_ceilings_are_fixed_and_ab_symmetric() -> None:
    proposal = {
        side: call_contract_for_side(
            side,
            service="proposal",
            prompt_digest="p" * 64,
            response_schema_digest="s" * 64,
        )
        for side in ("side_a", "side_b")
    }
    implementation = {
        side: call_contract_for_side(
            side,
            service="implementation",
            prompt_digest="i" * 64,
            response_schema_digest="r" * 64,
        )
        for side in ("side_a", "side_b")
    }
    assert PROPOSAL_TOKEN_CEILING == 6000
    assert IMPLEMENTATION_TOKEN_CEILING == 20_000
    assert proposal["side_a"] == proposal["side_b"]
    assert implementation["side_a"] == implementation["side_b"]
    assert proposal["side_a"]["token_ceiling"] == 6000
    assert implementation["side_a"]["token_ceiling"] == 20_000


def test_training_checkpoint_is_inside_real_worker_capability(
    tmp_path: Path,
) -> None:
    side_root = tmp_path / "side_a"
    result_root = side_root / "experiments/slot-01-matched-bpr/worker"
    checkpoint_root = result_root / "checkpoints"
    runtime_view = side_root / "experiments/slot-01-matched-bpr/runtime_view"
    for path in (side_root, result_root, checkpoint_root, runtime_view):
        path.mkdir(parents=True, exist_ok=True)
    capability = build_training_filesystem_capability(
        instance_private_root=side_root,
        result_root=result_root,
        checkpoint_root=checkpoint_root,
        project_root=runtime_view,
        recbole_root=RECBole_ROOT,
        dataset_root=SEARCH_DATASET_ROOT,
    )
    assert Path(capability.checkpoint_root).is_relative_to(
        Path(capability.result_root)
    )
