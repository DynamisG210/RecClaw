from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.mechanism_space import (  # noqa: E402
    CompileReportV1,
    CompileStatus,
    MechanismSpaceProviderV1,
    SpaceIdentity,
    available_space_ids,
    capability_envelope_digest,
    catalog_digest,
    compile_program,
    compile_program_bytes,
    prompt_projection,
    space_identity,
    validate_development_binding,
    validate_development_binding_bytes,
)
from recclaw_core.mechanism_space import kernel as kernel_module  # noqa: E402
from recclaw_core.search_spaces.bl_icf_v1 import provider as provider_module  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
SPACE_ID = "BL_ICF_MECHANISM_SPACE_V1"


def _fixture_document() -> dict[str, object]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def _fixture(name: str) -> dict[str, object]:
    for fixture in _fixture_document()["fixtures"]:
        if fixture["anchor_name"] == name:
            return fixture
    raise KeyError(name)


def _program(name: str = "BPR_MF") -> dict[str, object]:
    return copy.deepcopy(_fixture(name)["program"])


def _rename_component(program: dict[str, object], old: str, new: str) -> None:
    payload = program["program_payload"]
    for component in payload["components"]:
        if component["component_id"] == old:
            component["component_id"] = new
        for input_item in component["inputs"]:
            source = input_item["source"]
            if source["kind"] == "COMPONENT" and source["component_id"] == old:
                source["component_id"] = new
    for operator in payload["architecture_operators"]:
        operator["targets"] = [new if item == old else item for item in operator["targets"]]
        operator["replacements"] = [new if item == old else item for item in operator["replacements"]]
    for ablation in payload["ablation_plan"]:
        ablation["remove_component_ids"] = [
            new if item == old else item for item in ablation["remove_component_ids"]
        ]


def _custom_program() -> dict[str, object]:
    program = _program("BPR_MF")
    payload = program["program_payload"]
    payload["construction_mode"] = "CUSTOM_MODEL"
    payload["changed_slots"] = [{"slot_id": "ENCODER", "change_role": "CORE"}]
    payload["core_hypothesis"] = (
        "A from-scratch train-only collaborative encoder can replace inherited BPR and LightGCN encoders."
    )
    payload["components"][1] = {
        "component_id": "encoder",
        "slot_id": "ENCODER",
        "custom_component_id": "custom_encoder",
        "inputs": [
            {
                "port": "representation",
                "source": {
                    "kind": "COMPONENT",
                    "component_id": "embedding",
                    "output_port": "embedding",
                },
            }
        ],
        "parameters": {},
    }
    payload["custom_components"] = [
        {
            "custom_component_id": "custom_encoder",
            "slot_id": "ENCODER",
            "mathematical_definition": (
                "For every user and item, map the trainable ID embedding through a bounded residual operator R(E)."
            ),
            "algorithm_definition": (
                "Initialize ID embeddings, apply the declared residual operator, and expose user/item representations."
            ),
            "input_ports": [
                {
                    "port": "representation",
                    "types": ["bl_icf/embedding"],
                    "minimum": 1,
                }
            ],
            "output_ports": [
                {"port": "representation", "type": "bl_icf/representation"}
            ],
            "allowed_read_roles": ["USER_ID", "ITEM_ID", "TRAIN_INTERACTIONS"],
            "family_boundary_justification": (
                "The component reads only IDs and train interactions and emits the same relevance-score representation contract."
            ),
            "minimal_implementation": {
                "entrypoint_role": "MODEL",
                "file_roles": ["MODEL", "TEST"],
                "steps": ["Implement the residual encoder", "Add shape and gradient tests"],
            },
            "matched_control_rationale": (
                "Compare against the same embedding, score, sampler, objective, and training program using encoder.none_mf."
            ),
            "ablation": "Replace the residual operator with the identity map.",
            "failure_modes": ["The residual operator may collapse representations or add cost without ranking gain."],
            "estimated_cost": "MEDIUM",
        }
    ]
    payload["architecture_operators"] = [
        {
            "operator_id": "synthesize_custom_model",
            "targets": [],
            "replacements": ["encoder"],
            "parameters": {},
            "rationale": "Instantiate a complete candidate-local encoder without inheriting an anchor model class.",
        }
    ]
    payload["ablation_plan"][0]["remove_component_ids"] = ["encoder"]
    return program


def _binding_for(program: dict[str, object]) -> dict[str, object]:
    report = compile_program(program)
    assert report.is_valid
    assert report.space_identity is not None
    assert report.candidate_id is not None
    assert report.mechanism_program_digest is not None
    assert report.mechanism_semantics_digest is not None
    run_id = "bl-icf-fixture-run-001"
    binding = {
        "record_type": "CANDIDATE_EXECUTION_BINDING",
        "binding_version": "recclaw.candidate-execution-binding.v1",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
        "run_id": run_id,
        "candidate_id": report.candidate_id,
        "search_space_id": report.space_identity.search_space_id,
        "search_space_digest": report.space_identity.search_space_digest,
        "mechanism_program_digest": report.mechanism_program_digest,
        "mechanism_semantics_digest": report.mechanism_semantics_digest,
        "profile_ref": copy.deepcopy(program["profile_ref"]),
        "budget_ref": {"id": "fixture-budget-v1", "digest": "2" * 64},
        "implementation_ref": {"id": "fixture-implementation-v1", "digest": "3" * 64},
        "capability_envelope": {
            "envelope_id": "fixture-development-capabilities-v1",
            "envelope_digest": "0" * 64,
            "granted_capabilities": sorted(report.required_capabilities),
            "write_roots": [
                f"recclaw_ext/generated/{report.candidate_id}/",
                f"artifacts/{run_id}/",
            ],
        },
    }
    binding["capability_envelope"]["envelope_digest"] = capability_envelope_digest(
        binding["capability_envelope"]
    )
    return binding


class BL_ICFResourceTests(unittest.TestCase):
    def test_closed_catalog_exposes_only_registered_space(self) -> None:
        self.assertEqual(available_space_ids(), (SPACE_ID,))
        identity = space_identity(SPACE_ID)
        self.assertRegex(identity.search_space_digest, r"^[0-9a-f]{64}$")
        self.assertRegex(catalog_digest(), r"^[0-9a-f]{64}$")
        self.assertNotEqual(identity.search_space_digest, catalog_digest())
        closure = provider_module._load_resources().closure
        self.assertEqual(
            set(closure["kernel_source_sha256"]),
            {
                "recclaw_core/mechanism_space/canonical.py",
                "recclaw_core/mechanism_space/contracts.py",
                "recclaw_core/mechanism_space/kernel.py",
            },
        )
        self.assertEqual(
            dict(closure["runtime_dependencies"]),
            {"jsonschema": "4.26.0", "rfc8785": "0.1.4"},
        )

    def test_registry_closes_sixteen_axes_and_full_known_primitive_set(self) -> None:
        resources = provider_module._load_resources()
        self.assertEqual(set(resources.axes), {
            "RELATION_VIEW", "EMBEDDING", "ENCODER", "MESSAGE",
            "PROPAGATION_AGGREGATION", "RELATION_DECOMPOSITION",
            "FUSION_ROUTING", "SCORE_HEAD", "PRIMARY_OBJECTIVE",
            "NEGATIVE_SAMPLER", "SELF_SUPERVISION",
            "GEOMETRY_REGULARIZATION", "DENOISING_LONG_TAIL",
            "TRAINING_PROCEDURE", "EFFICIENCY_APPROXIMATION",
            "POSTHOC_RERANK",
        })
        self.assertGreaterEqual(len(resources.primitives), 200)
        self.assertEqual(
            set(resources.operators),
            {
                "add_component",
                "remove_component",
                "replace_component",
                "split_component",
                "fuse_components",
                "share_parameters",
                "decouple_relations",
                "route_by_user_or_item",
                "gate_component",
                "reweight_relation",
                "normalize_signal",
                "precompute_operator",
                "sparsify_relation",
                "approximate_fixed_point",
                "compile_propagation_to_constraint",
                "derive_closed_form",
                "distill_teacher",
                "prune_path",
                "freeze_path",
                "alternate_optimization",
                "synthesize_custom_model",
            },
        )
        self.assertTrue(
            {
                "RECBOLE_CORE_EDIT",
                "DATA_SPLIT_CHANGE",
                "EVALUATOR_CHANGE",
                "CANDIDATE_UNIVERSE_CHANGE",
                "BASELINE_RESULT_CHANGE",
                "FORMAL_CLAIM_CHANGE",
                "FORMAL_REGISTRY_CHANGE",
                "ONLINE_DEPENDENCY_INSTALL",
                "EXTERNAL_DATA_ACCESS",
                "OTHER_CANDIDATE_WRITE",
            }.issubset(set(resources.capability_policy["forbidden_changes"]))
        )
        allowed_capabilities = set(resources.capability_policy["capability_families"])
        for primitive_id, primitive in resources.primitives.items():
            with self.subTest(primitive=primitive_id):
                Draft202012Validator.check_schema(
                    provider_module.deep_thaw(primitive["parameter_schema"])
                )
                self.assertTrue(set(primitive["capabilities"]).issubset(allowed_capabilities))
                self.assertEqual(
                    len({item["port"] for item in primitive["output_ports"]}),
                    len(primitive["output_ports"]),
                )
                self.assertTrue(primitive["output_ports"])
        for operator_id, operator in resources.operators.items():
            with self.subTest(operator=operator_id):
                self.assertTrue(set(operator["capabilities"]).issubset(allowed_capabilities))
        with self.assertRaises(TypeError):
            resources.primitives["objective.bpr"]["slot_id"] = "ATTACKER_SLOT"
        with self.assertRaises(TypeError):
            resources.family["frozen_protocol_fields"] += ("attacker_field",)

    def test_excluded_primary_families_are_not_registered_as_primitives(self) -> None:
        resources = provider_module._load_resources()
        joined = "\n".join(resources.primitives).lower()
        for forbidden in ("admmslim", "slim", "ease", "sasrec", "bert4rec"):
            self.assertNotIn(forbidden, joined)

    def test_unknown_space_fails_closed(self) -> None:
        program = _program()
        program["search_space_id"] = "UNKNOWN_SPACE"
        report = compile_program(program)
        self.assertEqual(report.status, CompileStatus.UNSUPPORTED)
        self.assertEqual(report.diagnostics[0].code, "UNSUPPORTED_SEARCH_SPACE")

    def test_strict_json_rejects_duplicate_members_before_dispatch(self) -> None:
        report = compile_program_bytes(b'{"search_space_id":"first","search_space_id":"second"}')
        self.assertEqual(report.status, CompileStatus.INVALID)
        self.assertEqual(report.diagnostics[0].code, "JSON_DUPLICATE_KEY")
        program = json.dumps(_program(), separators=(",", ":")).encode("utf-8")
        duplicate_binding = b'{"record_type":"first","record_type":"second"}'
        report = validate_development_binding_bytes(program, duplicate_binding)
        self.assertEqual(report.status, CompileStatus.INVALID)
        self.assertEqual(report.diagnostics[0].code, "JSON_DUPLICATE_KEY")

    def test_provider_imports_directly_outside_repository_cwd(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)
        with tempfile.TemporaryDirectory() as outside_cwd:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from recclaw_core.search_spaces.bl_icf_v1.provider "
                        "import PROVIDER; print(PROVIDER.identity().search_space_id)"
                    ),
                ],
                cwd=outside_cwd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), SPACE_ID)

    def test_kernel_extension_contract_is_not_bl_icf_hardcoded(self) -> None:
        class SyntheticProvider:
            provider_id = "test.synthetic.provider.v1"

            def identity(self) -> SpaceIdentity:
                return SpaceIdentity(
                    "SYNTHETIC_TEST_SPACE_V1",
                    "1",
                    "a" * 64,
                    "SYNTHETIC_FAMILY_V1",
                    "1",
                    self.provider_id,
                )

            def compile(self, envelope: dict[str, object]) -> CompileReportV1:
                return CompileReportV1(CompileStatus.VALID_NEEDS_IMPLEMENTATION, space_identity=self.identity())

            def prompt_projection(self) -> dict[str, object]:
                return {"test_only": True}

            def program_schema(self) -> dict[str, object]:
                return {"type": "object"}

        envelope = {
            "record_type": "MECHANISM_PROGRAM_ENVELOPE",
            "kernel_schema_version": "recclaw.mechanism-space.kernel.v1",
            "search_space_id": "SYNTHETIC_TEST_SPACE_V1",
            "search_space_digest": "a" * 64,
            "family_id": "SYNTHETIC_FAMILY_V1",
            "family_version": "1",
            "profile_ref": {
                "profile_id": "test-profile",
                "profile_digest": "b" * 64,
                "profile_kind": "TEST_ONLY",
            },
            "program_payload": {},
        }
        provider = SyntheticProvider()
        self.assertIsInstance(provider, MechanismSpaceProviderV1)
        report = kernel_module._compile_with_provider(envelope, provider)
        self.assertEqual(report.status, CompileStatus.VALID_NEEDS_IMPLEMENTATION)
        self.assertEqual(available_space_ids(), (SPACE_ID,))


class BL_ICFAnchorExpressibilityTests(unittest.TestCase):
    def test_all_anchor_fixtures_compile_to_expected_status_and_primitives(self) -> None:
        document = _fixture_document()
        self.assertTrue(document["test_only"])
        self.assertEqual(document["proposal_prompt_exposure"], "FORBIDDEN")
        self.assertEqual(len(document["fixtures"]), 11)
        for fixture in document["fixtures"]:
            with self.subTest(anchor=fixture["anchor_name"]):
                report = compile_program(fixture["program"])
                self.assertEqual(report.status.value, fixture["expected_status"], report.to_json())
                self.assertIsNotNone(report.resolved_ir)
                observed = {
                    component["primitive_id"]
                    for component in report.resolved_ir["components"]
                    if "primitive_id" in component
                }
                self.assertEqual(observed, set(fixture["expected_primitives"]))

    def test_ultragcn_fixture_has_no_message_passing_and_explicit_constraints(self) -> None:
        report = compile_program(_program("ULTRAGCN"))
        self.assertEqual(report.status, CompileStatus.VALID_NEEDS_IMPLEMENTATION, report.to_json())
        primitives = {
            component.get("primitive_id")
            for component in report.resolved_ir["components"]
        }
        self.assertNotIn("encoder.explicit_message_passing", primitives)
        self.assertFalse(any(value and value.startswith("message.") for value in primitives))
        self.assertFalse(any(value and value.startswith("propagation.") for value in primitives))
        self.assertIn("encoder.fixed_point_constraint", primitives)
        self.assertIn("relation.item_item_cooccurrence_topk", primitives)
        self.assertIn("objective.user_item_structure_constraint", primitives)
        self.assertIn("objective.item_item_relation_constraint", primitives)
        self.assertIn("efficiency.remove_message_passing", primitives)
        operators = {item["operator_id"] for item in report.resolved_ir["architecture_operators"]}
        self.assertIn("compile_propagation_to_constraint", operators)

    def test_builtin_anchor_materialization_is_exact_and_unsupported_parameters_fall_back(self) -> None:
        bpr = compile_program(_program("BPR_MF"))
        self.assertEqual(bpr.status, CompileStatus.VALID_WIRED, bpr.to_json())
        self.assertEqual(
            bpr.resolved_ir["builtin_runtime_materialization"]["mechanism_overrides"],
            {
                "embedding_size": 64,
                "learning_rate": 0.001,
                "train_neg_sample_args": {
                    "distribution": "uniform",
                    "sample_num": 1,
                    "alpha": 1.0,
                    "dynamic": False,
                    "candidate_num": 0,
                },
            },
        )

        lightgcn = compile_program(_program("LIGHTGCN"))
        self.assertEqual(lightgcn.status, CompileStatus.VALID_WIRED, lightgcn.to_json())
        self.assertEqual(
            lightgcn.resolved_ir["builtin_runtime_materialization"]["mechanism_overrides"][
                "n_layers"
            ],
            3,
        )

        unsupported = _program("BPR_MF")
        objective = next(
            item
            for item in unsupported["program_payload"]["components"]
            if item["primitive_id"] == "objective.bpr"
        )
        objective["parameters"]["weight"] = 0.5
        report = compile_program(unsupported)
        self.assertEqual(report.status, CompileStatus.VALID_NEEDS_IMPLEMENTATION)
        self.assertIsNone(report.resolved_ir["builtin_runtime_materialization"])

    def test_custom_model_escape_hatch_does_not_require_anchor_inheritance(self) -> None:
        program = _custom_program()
        self.assertEqual(program["program_payload"]["parent_refs"], [])
        report = compile_program(program)
        self.assertEqual(report.status, CompileStatus.VALID_NEEDS_IMPLEMENTATION, report.to_json())
        self.assertIn("CUSTOM_MODEL_IMPLEMENTATION", report.required_capabilities)
        self.assertEqual(
            report.resolved_ir["candidate_root"],
            f"recclaw_ext/generated/{report.candidate_id}/",
        )

    def test_incomplete_custom_escape_hatch_is_rejected(self) -> None:
        program = _custom_program()
        del program["program_payload"]["custom_components"][0]["mathematical_definition"]
        report = compile_program(program)
        self.assertEqual(report.status, CompileStatus.INVALID)
        self.assertTrue(any(item.code == "BL_ICF_SCHEMA_INVALID" for item in report.diagnostics))


class BL_ICFIdentityAndBoundaryTests(unittest.TestCase):
    def test_narrative_and_opaque_rename_do_not_change_semantics_digest(self) -> None:
        original = _program()
        renamed = copy.deepcopy(original)
        _rename_component(renamed, "encoder", "renamed_encoder")
        renamed["program_payload"]["research_question"] = (
            "Does an opaque component label alter any actual mathematical or data-flow semantics?"
        )
        first = compile_program(original)
        second = compile_program(renamed)
        self.assertTrue(first.is_valid and second.is_valid)
        self.assertEqual(first.mechanism_semantics_digest, second.mechanism_semantics_digest)
        self.assertNotEqual(first.mechanism_program_digest, second.mechanism_program_digest)
        self.assertNotEqual(first.candidate_id, second.candidate_id)

    def test_material_score_change_changes_semantics_digest(self) -> None:
        original = _program()
        changed = copy.deepcopy(original)
        score_component = next(
            item for item in changed["program_payload"]["components"]
            if item["component_id"] == "score"
        )
        score_component["primitive_id"] = "score.cosine_similarity"
        first = compile_program(original)
        second = compile_program(changed)
        self.assertTrue(first.is_valid and second.is_valid)
        self.assertNotEqual(first.mechanism_semantics_digest, second.mechanism_semantics_digest)

    def test_training_sampler_change_is_mechanism_not_protocol_branch(self) -> None:
        program = _program()
        sampler_component = next(
            item for item in program["program_payload"]["components"]
            if item["component_id"] == "sampler"
        )
        sampler_component["parameters"]["negative_count"] = 8
        report = compile_program(program)
        self.assertTrue(report.is_valid, report.to_json())

    def test_frozen_protocol_change_is_routed_to_branch(self) -> None:
        program = _program()
        program["program_payload"]["protocol_impact"] = {
            "status": "BRANCH_REQUIRED",
            "requested_changes": ["split"],
        }
        report = compile_program(program)
        self.assertEqual(report.status, CompileStatus.PROTOCOL_BRANCH_REQUIRED, report.to_json())
        self.assertEqual(report.diagnostics[0].code, "PROTOCOL_BRANCH_REQUIRED")

    def test_external_or_test_data_role_is_rejected_before_compile(self) -> None:
        program = _program()
        program["program_payload"]["components"][0]["inputs"][0]["source"] = {
            "kind": "DATA",
            "data_role": "TEST_STATISTICS",
        }
        report = compile_program(program)
        self.assertEqual(report.status, CompileStatus.INVALID)

    def test_cross_family_primitive_is_rejected_without_bridge(self) -> None:
        program = _program()
        program["program_payload"]["components"][1]["primitive_id"] = (
            "sequential.transformer_encoder"
        )
        report = compile_program(program)
        self.assertEqual(report.status, CompileStatus.INVALID)
        self.assertTrue(any(item.code == "UNKNOWN_PRIMITIVE" for item in report.diagnostics))

    def test_typed_component_graph_rejects_missing_ports_bad_sources_and_cycles(self) -> None:
        missing_port = _program()
        score = next(
            item
            for item in missing_port["program_payload"]["components"]
            if item["component_id"] == "score"
        )
        score["inputs"] = []
        report = compile_program(missing_port)
        self.assertEqual(report.status, CompileStatus.INVALID)
        self.assertTrue(any(item.code == "REQUIRED_INPUT_MISSING" for item in report.diagnostics))

        bad_source = _program()
        score = next(
            item
            for item in bad_source["program_payload"]["components"]
            if item["component_id"] == "score"
        )
        score["inputs"][0]["source"]["output_port"] = "not_a_real_port"
        report = compile_program(bad_source)
        self.assertEqual(report.status, CompileStatus.INVALID)
        self.assertTrue(any(item.code == "UNKNOWN_SOURCE_PORT" for item in report.diagnostics))

        cyclic = _program()
        encoder = next(
            item
            for item in cyclic["program_payload"]["components"]
            if item["component_id"] == "encoder"
        )
        encoder["inputs"][0]["source"] = {
            "kind": "COMPONENT",
            "component_id": "score",
            "output_port": "score",
        }
        report = compile_program(cyclic)
        self.assertEqual(report.status, CompileStatus.INVALID)
        self.assertTrue(any(item.code == "COMPONENT_GRAPH_CYCLE" for item in report.diagnostics))

    def test_development_binding_requires_exact_identity_capabilities_and_roots(self) -> None:
        program = _custom_program()
        binding = _binding_for(program)
        accepted = validate_development_binding(program, binding)
        self.assertEqual(
            accepted.status,
            CompileStatus.VALID_NEEDS_IMPLEMENTATION,
            accepted.to_json(),
        )

        narrowed = copy.deepcopy(binding)
        narrowed["capability_envelope"]["granted_capabilities"].remove(
            "CUSTOM_MODEL_IMPLEMENTATION"
        )
        denied = validate_development_binding(program, narrowed)
        self.assertEqual(denied.status, CompileStatus.CAPABILITY_DENIED)
        self.assertTrue(any(item.code == "CAPABILITY_SET_MISMATCH" for item in denied.diagnostics))

        overgranted = copy.deepcopy(binding)
        overgranted["capability_envelope"]["granted_capabilities"].append("POSTHOC_RERANK")
        overgranted["capability_envelope"]["envelope_digest"] = capability_envelope_digest(
            overgranted["capability_envelope"]
        )
        denied = validate_development_binding(program, overgranted)
        self.assertEqual(denied.status, CompileStatus.CAPABILITY_DENIED)
        self.assertTrue(any(item.code == "CAPABILITY_SET_MISMATCH" for item in denied.diagnostics))

        tampered = copy.deepcopy(binding)
        tampered["capability_envelope"]["envelope_id"] = "tampered-envelope"
        denied = validate_development_binding(program, tampered)
        self.assertEqual(denied.status, CompileStatus.CAPABILITY_DENIED)
        self.assertTrue(
            any(item.code == "CAPABILITY_ENVELOPE_DIGEST_MISMATCH" for item in denied.diagnostics)
        )

        widened = copy.deepcopy(binding)
        widened["capability_envelope"]["write_roots"][1] = "recclaw_ext/models/"
        rejected = validate_development_binding(program, widened)
        self.assertEqual(rejected.status, CompileStatus.CAPABILITY_DENIED)
        self.assertTrue(any(item.code == "WRITE_ROOT_WIDENING" for item in rejected.diagnostics))

    def test_prompt_projection_contains_grammar_not_anchor_answer_recipes(self) -> None:
        projection = prompt_projection(SPACE_ID)
        text = json.dumps(projection, ensure_ascii=False, sort_keys=True).lower()
        self.assertIn("compile_propagation_to_constraint", text)
        self.assertIn("objective.bpr", text)
        for answer_name in (
            "sgl", "simgcl", "xsimgcl", "ncl", "lightgcl",
            "directau", "simplex", "ultragcn", "gf-cf",
        ):
            self.assertIsNone(
                re.search(rf"(?<![a-z0-9]){re.escape(answer_name)}(?![a-z0-9])", text),
                answer_name,
            )
        self.assertNotIn("anchor_name", text)
        self.assertNotIn("expected_primitives", text)


if __name__ == "__main__":
    unittest.main()
