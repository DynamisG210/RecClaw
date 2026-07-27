from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from jsonschema import Draft202012Validator
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.interaction import Interaction

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    CampaignRuntimeError,
    bl_icf_executable_profile_v2,
    campaign_projection,
    campaign_proposal_schema,
    campaign_readiness_failures,
    campaign_runtime_profile,
    executable_mechanisms,
    execution_recipe_for_program,
    program_from_proposal,
)
from recclaw_core.experiments.helix_abc_v1.campaign_dataset import (
    campaign_development_protocol,
)
from recclaw_core.experiments.helix_abc_v1.common_execution_guard import (
    CommonExecutionGuardV1,
)
from recclaw_core.experiments.helix_abc_v1.materialization import (
    DeterministicMaterializerV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import canary_budget
from recclaw_core.mechanism_space import compile_program
from recclaw_ext.models.composable_v2 import (
    BPRComposableV2,
    LightGCNComposableV2,
)


DATA_PATH = "/root/projects/RecBole/tests/test_data"


def _config(model: str, operators: list[str]) -> Config:
    config = Config(
        model=model,
        dataset="test",
        config_dict={
            "composition_operators": operators,
            "data_path": DATA_PATH,
            "device": "cpu",
            "embedding_size": 8,
            "item_inter_num_interval": "[0,inf)",
            "load_col": {"inter": ["user_id", "item_id"]},
            "n_layers": 2,
            "reg_weight": 0.0001,
            "use_gpu": False,
            "user_inter_num_interval": "[0,inf)",
        },
    )
    config["device"] = torch.device("cpu")
    return config


def _loss_backward(model) -> float:
    interaction = Interaction(
        {
            model.USER_ID: torch.tensor([1, 2]),
            model.ITEM_ID: torch.tensor([1, 2]),
            model.NEG_ITEM_ID: torch.tensor([3, 4]),
        }
    )
    loss = model.calculate_loss(interaction)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return float(loss.detach())


def test_profile_v2_is_main_grade_unique_and_content_bound() -> None:
    mechanisms = executable_mechanisms()
    profile = campaign_runtime_profile()
    assert profile["profile_id"] == "BL_ICF_EXECUTABLE_PROFILE_V2"
    assert profile["candidate_contract"] == "CandidateProposalV4"
    assert profile["executable_profile_digest"] == (
        bl_icf_executable_profile_v2()["profile_digest"]
    )
    assert len(mechanisms) == 66
    assert len({item.mechanism_program_digest for item in mechanisms}) == 66
    assert len({item.mechanism_semantics_digest for item in mechanisms}) == 66
    assert len({item.execution_recipe_digest for item in mechanisms}) == 66
    assert campaign_readiness_failures(import_entrypoints=True) == ()
    assert campaign_projection()["search_space_id"] == (
        "BL_ICF_EXECUTABLE_PROFILE_V2"
    )
    assert {
        len(item.operator_ids) for item in mechanisms
    } == {0, 1, 2}


def test_every_generated_composition_compiles_and_renders_exactly() -> None:
    guard = CommonExecutionGuardV1()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        for mechanism in executable_mechanisms():
            report = compile_program(mechanism.mechanism_program)
            assert report.is_valid
            recipe = execution_recipe_for_program(
                mechanism.mechanism_program
            )
            assert recipe["mechanism_id"] == mechanism.mechanism_id
            assert tuple(recipe["operator_ids"]) == mechanism.operator_ids
            decision, action = guard.plan_check(
                program=mechanism.mechanism_program,
                caller_compile_report=report,
                protocol=campaign_development_protocol(),
                budget=canary_budget(),
            )
            assert decision.decision == "COMMON_PASS"
            assert action is not None
            materialized = DeterministicMaterializerV1().materialize(
                action,
                program=mechanism.mechanism_program,
                arm_runtime_root=root,
            )
            assert materialized.candidate_id == mechanism.candidate_id


def test_typed_composition_resolves_without_recipe_selection() -> None:
    program = program_from_proposal(
        {
            "composition": {
                "base_mechanism_id": "LIGHTGCN",
                "primary_operator_id": "LGCN_DEBIASED_NEGATIVE",
                "secondary_operator_id": "LGCN_AUX_ALIGNMENT",
            }
        }
    )
    recipe = execution_recipe_for_program(program)
    assert recipe["operator_ids"] == [
        "LGCN_DEBIASED_NEGATIVE",
        "LGCN_AUX_ALIGNMENT",
    ]
    try:
        program_from_proposal(
            {
                "composition": {
                    "base_mechanism_id": "LIGHTGCN",
                    "primary_operator_id": "LGCN_SHALLOW",
                    "secondary_operator_id": "LGCN_EDGE_DROPOUT",
                }
            }
        )
    except CampaignRuntimeError:
        pass
    else:
        raise AssertionError("same-axis composition was accepted")


def test_provider_schema_requires_typed_composition_not_recipe_id() -> None:
    validator = Draft202012Validator(campaign_proposal_schema())
    proposal = {
        "candidate_label": "typed composition",
        "composition": {
            "base_mechanism_id": "BPR_MF",
            "primary_operator_id": "BPR_MIXED_NEGATIVE",
            "secondary_operator_id": "BPR_MARGIN",
        },
        "mechanism_hypothesis": "sampling and margin interact",
        "competing_hypothesis": "the interaction is neutral",
        "predicted_outcome_signature": "positive matched delta",
        "failure_mode": "no matched improvement",
        "parent_candidate_id": None,
        "proposal_intent": "DISCOVERY",
        "utility_features": {
            "useful_signal": 0.7,
            "frontier_potential": 0.7,
            "information_gain": 0.7,
        },
    }
    assert list(validator.iter_errors({"proposals": [proposal]})) == []
    legacy = dict(proposal)
    legacy.pop("composition")
    legacy["mechanism_id"] = "BPR_MIXED_NEGATIVE_MARGIN"
    assert list(validator.iter_errors({"proposals": [legacy]}))


def test_bpr_operator_families_execute_loss_and_backward() -> None:
    combinations = (
        ["BPR_MIXED_NEGATIVE", "BPR_MARGIN"],
        ["BPR_POPULARITY_NEGATIVE", "BPR_RANK_AWARE"],
        ["BPR_TAIL_REWEIGHT", "BPR_NORM_CONSTRAINT"],
        ["BPR_POPULARITY_REG"],
    )
    for operators in combinations:
        config = _config("BPR", operators)
        model = BPRComposableV2(config, create_dataset(config))
        _loss_backward(model)


def test_lightgcn_operator_families_execute_loss_and_backward() -> None:
    combinations = (
        ["LGCN_SHALLOW", "LGCN_LAYER_WEIGHTED"],
        ["LGCN_EDGE_DROPOUT", "LGCN_RESIDUAL"],
        ["LGCN_DEBIASED_NEGATIVE", "LGCN_RANK_AWARE"],
        ["LGCN_AUX_ALIGNMENT", "LGCN_NORM_CONSTRAINT"],
        ["LGCN_DUAL_PATH"],
    )
    for operators in combinations:
        config = _config("LightGCN", operators)
        model = LightGCNComposableV2(config, create_dataset(config))
        _loss_backward(model)
