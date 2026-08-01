"""Fail-closed Prefreeze V5 observability contract and validator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .canonical import bytes_sha256, canonical_json_bytes, canonical_value, sha256_digest
from .lab_api_broker import (
    LabApiBrokerReleaseV1,
    LabApiResponseFailureReasonV1,
)
from .prefreeze_v2 import (
    DOC_ROOT_REL,
    SQLITE_CALL_COLUMNS,
    V1_MANIFEST_REL,
    V4_ATTEMPT_RECEIPT_REL,
    V4_AUTH_REL,
    V4_BLOCKED_REL,
    V4_DRY_RUN_REL,
    V4_MANIFEST_REL,
    V4_NEGATIVE_FIXTURE_REL,
    V4_POLICY_REL,
    V4_PROVIDER_SCHEMA_REL,
    V4_RELEASE_REL,
    V4_VALIDATOR_REL,
    V4_VALID_FIXTURE_REL,
    exact_v4_probe_request_payload,
    verify_v1_seal,
    verify_v2_seal,
    verify_v3_seal,
)
from .wave2_integration import Wave2IntegrationError


V4_HEAD = "45aa25211322d9f765e28447d5d1e885ccf3ee70"
V4_PARENT = "68014e714d6636268fc36b259bf13498f158e4cd"
V4_TREE = "cf2b333b2930be78557b8e0502e6db995ae42aa7"

V5_HEAD = "d0fd84ce8174a4bfc913d98e1d0fa58e7480f363"
V5_PARENT = "45aa25211322d9f765e28447d5d1e885ccf3ee70"
V5_TREE = "72c0a36c1ebde8ffc80368cba624b4886e3fdf3d"

V4_SEALED_DIGESTS: dict[Path, str] = {
    V4_MANIFEST_REL: "932c72d875eb629fa617243244d72b64ad7c5d186fce45d46390a754a06e56dc",
    V4_POLICY_REL: "21f62cdb87e795ebfc611bfe10752bcc1f37e0220a7d5033e421434ee73e490f",
    V4_AUTH_REL: "599911d0b19833b7e5e9baad2c60b0b35f461fc98e4e22b3d0ff101021879323",
    V4_RELEASE_REL: "26eee7ac8e10130f3bdac88ef2e8975f9bf4394853d618c302d7b8dddf12378a",
    V4_DRY_RUN_REL: "c298ecb3a99ff3b8224bb92a19197b233af4d2db32fd2adf9951869e89a1b870",
    V4_ATTEMPT_RECEIPT_REL: "caf07471016017b76cce3bbc58ff4986b2485fe2b0c1cdd3dfe5f6c597c6df4a",
    V4_BLOCKED_REL: "2ab74bdcf79f23712d01fd95550a72596d1fd71be6ce9af8cdbee5e38c49c180",
    V4_PROVIDER_SCHEMA_REL: "1a03214b9e9b27036d0350398c3e8ab17fd6158c638aa75ad5354b041c3c7e5b",
    V4_VALID_FIXTURE_REL: "05cc80fa34c99145955dc41075b4f3cb9c33013f034cfcbafc1778ec4b742064",
    V4_NEGATIVE_FIXTURE_REL: "ff4a98243e0846189623061e3bbb8765450af386047ab2af23263867fc611ea8",
    V4_VALIDATOR_REL: "caa20a24cfea3e1ca45be3a5fb785a5e761ee6197b1c16bf87b491e261b08f9c",
}

V5_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v5-20260801"
V5_MANIFEST_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-attempt.v5"
V5_POLICY_SCHEMA = "recclaw.research-line.r1-provider-retry-policy.v5"
V5_AUTH_SCHEMA = "recclaw.research-line.prefreeze-v5-authorization.v1"
V5_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v5-provider-attempt-receipt.v1"
)
V5_DRY_RUN_SCHEMA = "recclaw.research-line.prefreeze-v5-dry-run-receipt.v1"
V5_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v5-blocked-receipt.v1"
V5_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v5"
V5_VERIFICATION_SCHEMA = (
    "recclaw.research-line.prefreeze-v5-verification-receipt.v1"
)

V5_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v5-observability-diagnostic-slot"
V5_SESSION_ID = (
    "fresh-open-spec-prefreeze-v5-observability-diagnostic-slot-session"
)
V5_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v5_private")

V5_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V5.json"
V5_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V5.json"
V5_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V5_AUTHORIZATION.json"
V5_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V5.json"
)
V5_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V5_BLOCKED_RECEIPT.json"
V5_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V5_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V5_DRY_RUN_RECEIPT.json"
V5_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V5_VERIFICATION_RECEIPT.json"
BROKER_SOURCE_REL = Path(
    "src/recclaw_core/experiments/helix_abc_v1/lab_api_broker.py"
)
V5_ACCEPTED_BROKER_SOURCE_DIGEST = (
    "3e255e71e128011e6533b259fcb3455da378b9ecb398862aed3d5eac8ae00441"
)

V5_SEALED_DIGESTS: dict[Path, str] = {
    V5_MANIFEST_REL: "e4b5468c2f0711cfc0b7d52330efa4240cc26bc41c0d0a45b8416bc01da3b936",
    V5_POLICY_REL: "5bfa9a9f11cf817f8f6c1a38392f514e0826665d93899256f9c7daf8b5120749",
    V5_AUTH_REL: "2ff14b4aa657743a3c76ad3738e20bc5e8f6e5b42f6f552f6b553c339b25af3d",
    V5_DRY_RUN_REL: "c6ee04e0ddb5e0636f861ca5c5dfe7b2f57e84b6bb283d49b84a26f62c86869f",
    V5_ATTEMPT_RECEIPT_REL: "8f6b74382d11b967d2b67f7771ad37b4bdb5c3a5e92446b885a44d1eb4779fed",
    V5_BLOCKED_REL: "5254d1d517f0f521d44e9fccf300f025f754cdab0d157a89d7ae7685b9ee8031",
}

V6_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v6-20260801"
V6_MANIFEST_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-attempt.v6"
V6_POLICY_SCHEMA = "recclaw.research-line.r1-provider-retry-policy.v6"
V6_AUTH_SCHEMA = "recclaw.research-line.prefreeze-v6-authorization.v1"
V6_RELEASE_SCHEMA = "recclaw.research-line.provider-release-contract.v6"
V6_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v6-provider-attempt-receipt.v1"
)
V6_DRY_RUN_SCHEMA = "recclaw.research-line.prefreeze-v6-dry-run-receipt.v1"
V6_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v6-blocked-receipt.v1"
V6_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v6"
V6_VERIFICATION_SCHEMA = (
    "recclaw.research-line.prefreeze-v6-verification-receipt.v1"
)

V6_REQUESTED_MODEL_ALIAS = "gpt-5.4"
V6_REQUIRED_RETURNED_SNAPSHOT = "gpt-5.4-2026-03-05"
V6_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v6-exact-model-pair-slot"
V6_SESSION_ID = "fresh-open-spec-prefreeze-v6-exact-model-pair-slot-session"
V6_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v6_private")

V6_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V6.json"
V6_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V6.json"
V6_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V6.json"
V6_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V6_AUTHORIZATION.json"
V6_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V6.json"
)
V6_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V6_BLOCKED_RECEIPT.json"
V6_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V6_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V6_DRY_RUN_RECEIPT.json"
V6_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V6_VERIFICATION_RECEIPT.json"

V6_HEAD = "6d10e44baf73824d6e69770e4632aea38eb79de4"
V6_PARENT = "d0fd84ce8174a4bfc913d98e1d0fa58e7480f363"
V6_TREE = "46666c6af830e2e4aafdea8e17be0a99bdf2ee6c"
V6_SEALED_DIGESTS: dict[Path, str] = {
    V6_RELEASE_REL: "d480d615096e5a62bfa22f89407ef160012657df3230a0aeade9401ce2453f44",
    V6_POLICY_REL: "58e6cf43c7c85f08cdefa0d05ee9e71526db0e9fa239dae4f86bfc05ea294194",
    V6_MANIFEST_REL: "b9afcfbe38f3f2a5ecb013f477a0181776dd9314739868b761843d53a7ff57f2",
    V6_AUTH_REL: "3e0d884569836a84678f12800ef8105e2450f60a70a4d25c996cbe5e99529907",
    V6_DRY_RUN_REL: "3f0f5bee76266b71a49ebcfd3583e6c49277a0ddc0fdd6fa189d75d74116e6a5",
    V6_ATTEMPT_RECEIPT_REL: "c721e773ee0e06f49d62ad7c3d91ab4208a1c8c9b10eb44fc6f404ad5198a801",
    V6_BLOCKED_REL: "5072e2b73e2e3e53e9546a3cf56970fe0e6f360aa43e91d29ed339c38d2e25b4",
}

V7_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v7-20260801"
V7_DIAGNOSTIC_TOKEN_CEILING = 6000
V7_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v7-provider-attempt-receipt.v1"
)
V7_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v7-blocked-receipt.v1"
V7_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v7"
V7_VERIFICATION_SCHEMA = (
    "recclaw.research-line.prefreeze-v7-verification-receipt.v1"
)
V7_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v7-ceiling-alignment-slot"
V7_SESSION_ID = "fresh-open-spec-prefreeze-v7-ceiling-alignment-slot-session"
V7_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v7_private")
V7_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V7.json"
V7_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V7.json"
V7_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V7.json"
V7_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V7_AUTHORIZATION.json"
V7_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V7.json"
)
V7_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V7_BLOCKED_RECEIPT.json"
V7_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V7_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V7_DRY_RUN_RECEIPT.json"
V7_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V7_VERIFICATION_RECEIPT.json"

V7_HEAD = "f6c376410162921f60ee1025c1c2703c5d423768"
V7_PARENT = "6d10e44baf73824d6e69770e4632aea38eb79de4"
V7_TREE = "fad96fc6ca0315da787aeea40de657189a0bb12a"
V7_SEALED_DIGESTS: dict[Path, str] = {
    V7_RELEASE_REL: "98df1e6aeb165dda576a9550a1582b62c53ef21135dc27fc577d0e8b1e64ffba",
    V7_POLICY_REL: "2da21f872d4edc97a179dfb56e494c98d60be2f48188eab9d1cb3deacd1418c2",
    V7_MANIFEST_REL: "e006f409810994f360a7ffcb2e0a40a9c6a206ee9853c635a10dea6cf0e7f01e",
    V7_AUTH_REL: "e61b156963af5febab5cebb85c8ea7e36234385020c23a33a275ea64d725939c",
    V7_DRY_RUN_REL: "229478a68cbbe20ca63c901b40edd20e8be58f094b4e516b3ceee936e272f794",
    V7_ATTEMPT_RECEIPT_REL: "412611f8a6d0a63ac7db7524b31fbd32aba2dc042a887edab845bd6bd3b134d8",
    V7_BLOCKED_REL: "c2f41adfd99cbf50b75de33bea21374c5897af0e180a14afb35440a8d02caa88",
}

V8_MODEL_SNAPSHOT = "gpt-5.4-2026-03-05"
V8_DIAGNOSTIC_TOKEN_CEILING = 6000
V8_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v8-20260801"
V8_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v8-exact-snapshot-slot"
V8_SESSION_ID = "fresh-open-spec-prefreeze-v8-exact-snapshot-slot-session"
V8_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v8_private")
V8_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V8.json"
V8_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V8.json"
V8_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V8.json"
V8_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V8_AUTHORIZATION.json"
V8_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V8.json"
)
V8_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V8_BLOCKED_RECEIPT.json"
V8_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V8_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V8_DRY_RUN_RECEIPT.json"
V8_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V8_VERIFICATION_RECEIPT.json"
V8_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v8-provider-attempt-receipt.v1"
)
V8_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v8-blocked-receipt.v1"
V8_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v8"
V8_VERIFICATION_SCHEMA = (
    "recclaw.research-line.prefreeze-v8-verification-receipt.v1"
)

V8_HEAD = "a076f82a94659bf5b6a1c271bdd10badbc29444a"
V8_PARENT = "f6c376410162921f60ee1025c1c2703c5d423768"
V8_TREE = "04c1ee1a9994d7f1c8350df8e3cda1c6d4ebb1ab"
V8_SEALED_DIGESTS: dict[Path, str] = {
    V8_RELEASE_REL: "f23e1fca21cc29398308dd2a9accced9ba2c2577c32fbb055507a6fc48bc4ecc",
    V8_POLICY_REL: "c5ec452cf72c2638c4d096c7135edb306afba8fbe21f070d3937c4c0c20dc339",
    V8_MANIFEST_REL: "b2b1951f57c708f4433900ccb653a1732f45d39dae73efd0639112b57f0da3eb",
    V8_AUTH_REL: "af5e64dfe8dfa606ff274acb2fe0dc613799c547a8d76c5ba51d77b7ab2c8942",
    V8_DRY_RUN_REL: "cf8aa720f0be97acb08ff8276b0a8e6bf239ccf1816080859d41551435053db0",
    V8_ATTEMPT_RECEIPT_REL: "aa6bca177806cd1b918817b14948b2d4beb0017a4214c87f6f5c6378e93784d2",
    V8_BLOCKED_REL: "27a948e5a6ffd0be68e009d5f913fae646ffa4649c51b97af91a054d449b0330",
}

V9_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v9-20260801"
V9_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v9-availability-slot"
V9_SESSION_ID = "fresh-open-spec-prefreeze-v9-availability-slot-session"
V9_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v9_private")
V9_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V9.json"
V9_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V9.json"
V9_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V9.json"
V9_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V9_AUTHORIZATION.json"
V9_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V9.json"
)
V9_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V9_BLOCKED_RECEIPT.json"
V9_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V9_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V9_DRY_RUN_RECEIPT.json"
V9_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V9_VERIFICATION_RECEIPT.json"

V9_HEAD = "042dc4d3d42595d768425d7e5dc53a05faa10ef1"
V9_PARENT = "a076f82a94659bf5b6a1c271bdd10badbc29444a"
V9_TREE = "5256c751f70190a91c01f9faa6fa2d46253df963"
V9_SEALED_DIGESTS: dict[Path, str] = {
    V9_RELEASE_REL: "1ce4fa40e18c83c6a234d2a4937aa71c8e4fb9eda4e0b07c09839402877554e7",
    V9_POLICY_REL: "aab4de895067f3a3522d1a3a8de953a739f154949eeae2e96e7ab8e2f5a9c5a7",
    V9_MANIFEST_REL: "40538791e7e69f7905529dcd82fcfc92ced372aa168117ab245e63d500323600",
    V9_AUTH_REL: "ae0ebca878c8b35235d35036fff68497286ce9a753af406bc24d0b1b9e940493",
    V9_DRY_RUN_REL: "a720d03ce0f98be7eb0232928ce8dc7136395c951cc095517fdd68715844393f",
    V9_ATTEMPT_RECEIPT_REL: "ff7a6d22120e2a540f2d61e286e94f21b0eb5c05b6455064fceeda2799fe6c99",
    V9_BLOCKED_REL: "74b7f28003da886ed698aac3505c7a2412922942c437a9cf6523265fd4c1cd3e",
}

V10_REQUEST_MODEL_ALIAS = "gpt-5.4"
V10_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v10-20260801"
V10_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v10-model-routing-slot"
V10_SESSION_ID = "fresh-open-spec-prefreeze-v10-model-routing-slot-session"
V10_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v10_private")
V10_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V10.json"
V10_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V10.json"
V10_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V10.json"
V10_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V10_AUTHORIZATION.json"
V10_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V10.json"
)
V10_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V10_BLOCKED_RECEIPT.json"
V10_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V10_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V10_DRY_RUN_RECEIPT.json"
V10_VERIFICATION_REL = (
    DOC_ROOT_REL / "PREFREEZE_V10_VERIFICATION_RECEIPT.json"
)

V10_HEAD = "1af05904230485f82b503adc051ae3b1b6833b47"
V10_PARENT = "042dc4d3d42595d768425d7e5dc53a05faa10ef1"
V10_TREE = "536ce93fa1649075006abc07dd9a06b0a551a662"
V10_SEALED_DIGESTS: dict[Path, str] = {
    V10_RELEASE_REL: "ee247da7f36ff3d2343372824119ee271cbbfa1e645c488c3d6ee2c678f0a663",
    V10_POLICY_REL: "73ab615411fbf78a56f92f14a827322ec4d8e2f6a768bfea34c529dc859f68c6",
    V10_MANIFEST_REL: "0ac93391930fbd751ea23435f37b06d3faa54bd9e3ee8e2bccb82d67b50aaa59",
    V10_AUTH_REL: "d7ef5194bbe7ad05e4728a44be9a62b394d816fdc6537296f13a03521040fb63",
    V10_DRY_RUN_REL: "fc1889beda4d04585abf450e6ad1a561cf00b0ea7b5b043bfa5d26f7ab8a89f9",
    V10_ATTEMPT_RECEIPT_REL: "90a6751835f1f3a46409fe3b5d894f579ebc4d980829f77e41c62baaa6618189",
    V10_BLOCKED_REL: "0801cde765050c881f3995d1d5ab7609696273aaf20351c263722f6e65b8eed4",
}

V11_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v11-20260801"
V11_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v11-engineering-validation-slot"
V11_SESSION_ID = "fresh-open-spec-prefreeze-v11-engineering-validation-session"
V11_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v11_private")
V11_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V11.json"
V11_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V11.json"
V11_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V11.json"
V11_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V11_AUTHORIZATION.json"
V11_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V11.json"
)
V11_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V11_BLOCKED_RECEIPT.json"
V11_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V11_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V11_DRY_RUN_RECEIPT.json"
V11_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V11_VERIFICATION_RECEIPT.json"


def _repo_ref(relative: Path) -> str:
    return relative.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise Wave2IntegrationError(f"JSON resource is not an object: {path}")
    if canonical_json_bytes(value) != path.read_bytes():
        raise Wave2IntegrationError(f"JSON resource is not canonical: {path}")
    return value


def _load_exact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _read_json(path)
    if canonical_value(observed) != canonical_value(expected):
        raise Wave2IntegrationError(f"fail-closed artifact mismatch: {path}")
    return observed


def v5_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V5 physical attempt ordinal must be 1..3")
    return V5_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v4_seal(repo_root: Path) -> dict[str, str]:
    verify_v1_seal(repo_root)
    verify_v2_seal(repo_root)
    verify_v3_seal(repo_root)
    for relative, digest in V4_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V4 bytes changed: {relative.as_posix()}"
            )
    return {
        relative.as_posix(): digest
        for relative, digest in V4_SEALED_DIGESTS.items()
    }


def diagnostic_reason_vocabulary(
    *, include_returned_model_identity: bool = False
) -> list[dict[str, str]]:
    stages = {
        "ENVELOPE_JSON_DECODE": "HTTP200_ENVELOPE_DECODE",
        "ENVELOPE_SHAPE": "HTTP200_ENVELOPE_SHAPE",
        "CHOICES_SHAPE": "HTTP200_CHOICES_SHAPE",
        "CHOICE_SHAPE": "HTTP200_CHOICE_SHAPE",
        "MESSAGE_SHAPE": "HTTP200_MESSAGE_SHAPE",
        "MESSAGE_CONTENT_TYPE_OR_EMPTY": "HTTP200_MESSAGE_CONTENT",
        "CONTENT_JSON_DECODE": "HTTP200_CONTENT_JSON_DECODE",
        "SCHEMA_VALIDATION": "HTTP200_STRICT_SCHEMA_VALIDATION",
        "PROPOSAL_COUNT": "HTTP200_PROPOSAL_COUNT",
        "USAGE_SHAPE": "HTTP200_USAGE_SHAPE",
        "TOKEN_USAGE_TYPE": "HTTP200_TOKEN_USAGE_TYPE",
        "TOKEN_CEILING": "HTTP200_TOKEN_CEILING",
    }
    if include_returned_model_identity:
        stages["RETURNED_MODEL_TYPE_OR_EMPTY"] = (
            "HTTP200_RETURNED_MODEL_IDENTITY"
        )
    enum_values = {item.value for item in LabApiResponseFailureReasonV1}
    expected_enum_values = set(stages)
    if not include_returned_model_identity:
        expected_enum_values.add("RETURNED_MODEL_TYPE_OR_EMPTY")
    if expected_enum_values != enum_values:
        raise Wave2IntegrationError("V5 reason vocabulary differs from Broker enum")
    return [
        {
            "reason_code": reason,
            "stage": stages[reason],
            "retry_eligible": "FALSE_TERMINAL_RESPONSE_CONTRACT",
        }
        for reason in sorted(stages)
    ]


def expected_v5_retry_policy(repo_root: Path) -> dict[str, Any]:
    v4_policy = _read_json(repo_root / V4_POLICY_REL)
    return {
        "schema": V5_POLICY_SCHEMA,
        "inherited_v4_policy_ref": _repo_ref(V4_POLICY_REL),
        "inherited_v4_policy_digest": V4_SEALED_DIGESTS[V4_POLICY_REL],
        "inherited_v4_policy_semantics_digest": sha256_digest(v4_policy),
        "proposal_slots_per_side": 8,
        "proposal_denominator_per_side": 8,
        "diagnostic_slot": {
            **deepcopy(v4_policy["diagnostic_slot"]),
            "slot_id": "PREFREEZE_V5_OBSERVABILITY_DIAGNOSTIC",
        },
        "response_contract_reason_vocabulary": diagnostic_reason_vocabulary(),
        "response_contract_failure": {
            "retry_eligible": False,
            "manual_patch": "FORBIDDEN",
            "successful_response_selection": "FORBIDDEN",
            "schema_relaxation": "FORBIDDEN",
            "candidate_admission": "FORBIDDEN",
            "mechanism_negative_evidence": False,
        },
        "missingness": deepcopy(v4_policy["missingness"]),
    }


def expected_prefreeze_v5_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v4_seal(repo_root)
    v4_manifest = _read_json(repo_root / V4_MANIFEST_REL)
    policy = expected_v5_retry_policy(repo_root)
    exact_contract = deepcopy(v4_manifest["exact_provider_contract"])
    exact_contract.update(
        {
            "logical_call_id": V5_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V5_SESSION_ID,
            "diagnostic_slot_id": "PREFREEZE_V5_OBSERVABILITY_DIAGNOSTIC",
        }
    )
    physical_identities = [
        {
            "ordinal": ordinal,
            "physical_attempt_identity_digest": sha256_digest(
                {
                    "attempt_id": V5_ATTEMPT_ID,
                    "diagnostic_slot": "PREFREEZE_V5_OBSERVABILITY_DIAGNOSTIC",
                    "ordinal": ordinal,
                    "private_root_digest": sha256_digest(
                        {"path": v5_physical_root(ordinal).as_posix()}
                    ),
                }
            ),
            "private_root_digest": sha256_digest(
                {"path": v5_physical_root(ordinal).as_posix()}
            ),
        }
        for ordinal in (1, 2, 3)
    ]
    future_contract = deepcopy(v4_manifest["future_r1_scientific_contract"])
    return {
        "schema": V5_MANIFEST_SCHEMA,
        "attempt_identity": {
            "attempt_id": V5_ATTEMPT_ID,
            "base_commit": V4_HEAD,
            "base_parent": V4_PARENT,
            "base_tree": V4_TREE,
            "pre_outcome": True,
            "distinct_from_v1_v2_v3_v4_attempts": True,
            "old_attempt_call_session_db_identity_reuse": False,
        },
        "sealed_predecessor_evidence": {
            "v4_manifest_digest": V4_SEALED_DIGESTS[V4_MANIFEST_REL],
            "v4_attempt_digest": V4_SEALED_DIGESTS[V4_ATTEMPT_RECEIPT_REL],
            "v4_blocked_digest": V4_SEALED_DIGESTS[V4_BLOCKED_REL],
            "v4_provider_schema_digest": V4_SEALED_DIGESTS[V4_PROVIDER_SCHEMA_REL],
            "v4_local_validator_digest": V4_SEALED_DIGESTS[V4_VALIDATOR_REL],
            "v1_v2_v3_v4_preservation": (
                "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
            ),
        },
        "engineering_observability_change": {
            "only_change": "ALLOWLISTED_CONTENT_FREE_HTTP200_PARSE_REASON",
            "broker_source_ref": _repo_ref(BROKER_SOURCE_REL),
            "broker_source_digest": V5_ACCEPTED_BROKER_SOURCE_DIGEST,
            "reason_vocabulary": diagnostic_reason_vocabulary(),
            "reason_vocabulary_digest": sha256_digest(
                diagnostic_reason_vocabulary()
            ),
            "sqlite_schema_change": False,
            "persistence_field": "calls.error_detail_json.reason_code",
            "raw_response_persisted_on_failure": False,
            "provider_body_persisted_on_failure": False,
            "exception_text_persisted": False,
            "prompt_secret_header_endpoint_literal_persisted": False,
            "success_path_changed": False,
            "schema_validation_changed": False,
            "token_accounting_changed": False,
            "provider_release_semantics_changed": False,
            "retry_eligibility_changed": False,
        },
        "exact_provider_contract": exact_contract,
        "response_contract_equivalence": deepcopy(
            v4_manifest["response_contract_equivalence"]
        ),
        "bounded_retry": {
            "policy_ref": _repo_ref(V5_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "maximum_physical_attempts": 3,
            "maximum_retry_count": 2,
            "deterministic_backoff_ms": [1000, 3000],
            "physical_attempt_identities": physical_identities,
            "sqlite_calls_schema_digest": sha256_digest(
                {"table": "calls", "columns": list(SQLITE_CALL_COLUMNS)}
            ),
        },
        "future_r1_scientific_contract": future_contract,
        "v4_scientific_contract_digest": sha256_digest(
            v4_manifest["future_r1_scientific_contract"]
        ),
        "v5_scientific_contract_digest": sha256_digest(future_contract),
        "pre_outcome_counters": {
            "provider_calls": 0,
            "research_candidates_generated": 0,
            "open_specs_projected": 0,
            "resolver_calls": 0,
            "candidate_roots_created": 0,
            "candidate_qualifications": 0,
            "candidate_admissions": 0,
            "training_runs": 0,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
        },
        "r1_worker_launch_authorized": False,
    }


def expected_v5_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v5_manifest(repo_root)
    policy = expected_v5_retry_policy(repo_root)
    return {
        "schema": V5_AUTH_SCHEMA,
        "status": "AUTHORIZED_ONE_V5_OBSERVABILITY_DIAGNOSTIC_SLOT",
        "attempt_id": V5_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V5_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V5_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "broker_source_digest": manifest["engineering_observability_change"][
            "broker_source_digest"
        ],
        "reason_vocabulary_digest": manifest[
            "engineering_observability_change"
        ]["reason_vocabulary_digest"],
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v5(repo_root: Path) -> dict[str, Any]:
    verify_v4_seal(repo_root)
    policy = _load_exact(
        repo_root / V5_POLICY_REL,
        expected_v5_retry_policy(repo_root),
    )
    manifest = _load_exact(
        repo_root / V5_MANIFEST_REL,
        expected_prefreeze_v5_manifest(repo_root),
    )
    _load_exact(
        repo_root / V5_AUTH_REL,
        expected_v5_authorization(repo_root),
    )
    v4_manifest = _read_json(repo_root / V4_MANIFEST_REL)
    v4_contract = v4_manifest["exact_provider_contract"]
    v5_contract = manifest["exact_provider_contract"]
    identity_fields = {
        "logical_call_id",
        "proposal_generation_session_id",
        "diagnostic_slot_id",
    }
    if {
        key: value for key, value in v5_contract.items() if key not in identity_fields
    } != {
        key: value for key, value in v4_contract.items() if key not in identity_fields
    }:
        raise Wave2IntegrationError("V5 changed the exact V4 Provider payload contract")
    if (
        manifest["future_r1_scientific_contract"]
        != v4_manifest["future_r1_scientific_contract"]
        or manifest["v4_scientific_contract_digest"]
        != manifest["v5_scientific_contract_digest"]
    ):
        raise Wave2IntegrationError("V5 changed the frozen R1 scientific contract")
    future = manifest["future_r1_scientific_contract"]
    shared = future["shared_call_contract_digest"]
    if (
        future["side_a_call_contract_digest"] != shared
        or future["side_b_call_contract_digest"] != shared
        or future["proposal_slots_per_side"] != 8
        or future["proposal_denominator_per_side"] != 8
    ):
        raise Wave2IntegrationError("V5 A/B symmetry or denominator changed")
    if policy["response_contract_reason_vocabulary"] != diagnostic_reason_vocabulary():
        raise Wave2IntegrationError("V5 policy reason vocabulary changed")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V5 is not pre-outcome")
    return manifest


def provider_free_v5_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v5(repo_root)
    return {
        "schema": V5_DRY_RUN_SCHEMA,
        "status": "PASS_PROVIDER_FREE_V5_OBSERVABILITY",
        "attempt_id": V5_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_v3_v4_seals_verified": True,
        "exact_v4_payload_contract_verified": True,
        "v4_response_semantics_verified_unchanged": True,
        "ab_call_contract_symmetry_verified": True,
        "allowlisted_reason_vocabulary_verified": True,
        "sqlite_schema_unchanged": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def v6_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V6 physical attempt ordinal must be 1..3")
    return V6_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v5_seal(repo_root: Path) -> dict[str, str]:
    """Prove that V1--V5 accepted evidence remains byte-identical."""

    verify_v4_seal(repo_root)
    for relative, digest in V5_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V5 bytes changed: {relative.as_posix()}"
            )
    accepted_broker_digest = _read_json(repo_root / V5_MANIFEST_REL)[
        "engineering_observability_change"
    ]["broker_source_digest"]
    if (
        accepted_broker_digest != V5_ACCEPTED_BROKER_SOURCE_DIGEST
    ):
        raise Wave2IntegrationError("accepted V5 Broker source identity changed")
    return {
        relative.as_posix(): digest
        for relative, digest in V5_SEALED_DIGESTS.items()
    }


def validate_v6_exact_model_pair(
    *, requested_model_alias: str, returned_model: str
) -> None:
    """Accept only the two exact, pre-frozen model identity literals."""

    if requested_model_alias != V6_REQUESTED_MODEL_ALIAS:
        raise Wave2IntegrationError("V6 requested model alias mismatch")
    if returned_model != V6_REQUIRED_RETURNED_SNAPSHOT:
        raise Wave2IntegrationError("V6 returned model snapshot mismatch")


def _v6_preserved_scientific_identity(repo_root: Path) -> dict[str, Any]:
    v1 = _read_json(repo_root / V1_MANIFEST_REL)
    preserved = {
        key: deepcopy(v1[key])
        for key in (
            "open_spec_contract",
            "r1_identity",
            "r2_identity",
            "runtime_identity",
            "evidence_policy",
            "shared_implementation",
        )
    }
    return {
        "source_manifest_ref": _repo_ref(V1_MANIFEST_REL),
        "source_manifest_digest": bytes_sha256(
            (repo_root / V1_MANIFEST_REL).read_bytes()
        ),
        "preserved_fields": sorted(preserved),
        "preserved_fields_digest": sha256_digest(preserved),
        "seed_root_db_candidate_package_outcome_memory_lineage_unchanged": True,
        "runtime_dependency_missingness_gate_threshold_analysis_unchanged": True,
        "shared_origin_blind_implementer_qualifier_unchanged": True,
        "manual_patch_prohibition_unchanged": True,
    }


def expected_v6_provider_release(repo_root: Path) -> dict[str, Any]:
    """Return the exact-pair release binding without changing transport bytes."""

    verify_v5_seal(repo_root)
    v5 = _read_json(repo_root / V5_MANIFEST_REL)
    v1 = _read_json(repo_root / V1_MANIFEST_REL)
    exact = v5["exact_provider_contract"]
    proposal = v1["shared_proposal_call"]
    preimage = {
        "schema": V6_RELEASE_SCHEMA,
        "transport_release_ref": exact["provider_release_ref"],
        "transport_release_artifact_digest": V4_SEALED_DIGESTS[V4_RELEASE_REL],
        "transport_release_contract_digest": exact["provider_release_digest"],
        "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
        "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
        "endpoint_digest": exact["endpoint_digest"],
        "credential_config_digest": exact["credential_config_digest"],
        "credential_identity_digest": exact["credential_identity_digest"],
        "response_schema_digest": exact["response_schema_digest"],
        "local_uniqueness_contract_digest": v5[
            "response_contract_equivalence"
        ]["local_uniqueness_contract_digest"],
        "prompt_digest": exact["prompt_digest"],
        "tool_policy_digest": exact["tool_policy_digest"],
        "request_mode": exact["request_mode"],
        "temperature": exact["temperature"],
        "diagnostic_slot_call_count": 1,
        "diagnostic_expected_proposals": 1,
        "diagnostic_token_budget": exact["token_budget"],
        "transport_max_total_tokens_per_call": proposal["token_budget"],
        "future_r1_call_count_per_side": proposal["call_count"],
        "future_r1_proposal_budget_per_side": proposal[
            "proposal_budget_per_side"
        ],
        "future_r1_expected_proposals_per_slot": proposal[
            "expected_proposals_per_call"
        ],
        "future_r1_token_budget_per_call": proposal["token_budget"],
        "alias_or_snapshot_matching": "EXACT_LITERAL_ONLY",
        "prefix_regex_startswith_or_arbitrary_snapshot": "FORBIDDEN",
        "silent_alias_fallback_model_or_endpoint_change": "FORBIDDEN",
    }
    return {**preimage, "release_digest": sha256_digest(preimage)}


def expected_v6_retry_policy(repo_root: Path) -> dict[str, Any]:
    policy = deepcopy(_read_json(repo_root / V5_POLICY_REL))
    policy["schema"] = V6_POLICY_SCHEMA
    policy["inherited_v5_policy_ref"] = _repo_ref(V5_POLICY_REL)
    policy["inherited_v5_policy_digest"] = V5_SEALED_DIGESTS[V5_POLICY_REL]
    policy.pop("inherited_v4_policy_ref", None)
    policy.pop("inherited_v4_policy_digest", None)
    policy.pop("inherited_v4_policy_semantics_digest", None)
    policy["diagnostic_slot"]["slot_id"] = "PREFREEZE_V6_EXACT_MODEL_PAIR"
    policy["exact_model_pair_failure"] = {
        "failure_class": "RETURNED_MODEL_PAIR_MISMATCH",
        "retry_eligible": False,
        "manual_patch": "FORBIDDEN",
        "successful_response_selection": "FORBIDDEN",
        "candidate_admission": "FORBIDDEN",
        "mechanism_negative_evidence": False,
    }
    return policy


def expected_prefreeze_v6_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v5_seal(repo_root)
    v5 = _read_json(repo_root / V5_MANIFEST_REL)
    release = expected_v6_provider_release(repo_root)
    policy = expected_v6_retry_policy(repo_root)
    exact = deepcopy(v5["exact_provider_contract"])
    exact.update(
        {
            "model": V6_REQUESTED_MODEL_ALIAS,
            "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
            "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
            "model_identity_pair_digest": sha256_digest(
                {
                    "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
                    "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
                }
            ),
            "transport_provider_release_digest": exact[
                "provider_release_digest"
            ],
            "provider_release_ref": _repo_ref(V6_RELEASE_REL),
            "provider_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(release)
            ),
            "provider_release_digest": release["release_digest"],
            "logical_call_id": V6_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V6_SESSION_ID,
            "diagnostic_slot_id": "PREFREEZE_V6_EXACT_MODEL_PAIR",
        }
    )
    physical_identities = [
        {
            "ordinal": ordinal,
            "physical_attempt_identity_digest": sha256_digest(
                {
                    "attempt_id": V6_ATTEMPT_ID,
                    "diagnostic_slot": "PREFREEZE_V6_EXACT_MODEL_PAIR",
                    "ordinal": ordinal,
                    "private_root_digest": sha256_digest(
                        {"path": v6_physical_root(ordinal).as_posix()}
                    ),
                }
            ),
            "private_root_digest": sha256_digest(
                {"path": v6_physical_root(ordinal).as_posix()}
            ),
        }
        for ordinal in (1, 2, 3)
    ]
    future = deepcopy(v5["future_r1_scientific_contract"])
    shared_call_digest = sha256_digest(
        {
            "inherited_v5_shared_call_contract_digest": future[
                "shared_call_contract_digest"
            ],
            "v6_provider_release_contract_digest": release["release_digest"],
        }
    )
    future.update(
        {
            "shared_call_contract_digest": shared_call_digest,
            "side_a_call_contract_digest": shared_call_digest,
            "side_b_call_contract_digest": shared_call_digest,
        }
    )
    return {
        "schema": V6_MANIFEST_SCHEMA,
        "attempt_identity": {
            "attempt_id": V6_ATTEMPT_ID,
            "base_commit": V5_HEAD,
            "base_parent": V5_PARENT,
            "base_tree": V5_TREE,
            "pre_outcome": True,
            "distinct_from_v1_v2_v3_v4_v5_attempts": True,
            "old_attempt_call_session_db_identity_reuse": False,
        },
        "sealed_predecessor_evidence": {
            "v5_manifest_digest": V5_SEALED_DIGESTS[V5_MANIFEST_REL],
            "v5_attempt_digest": V5_SEALED_DIGESTS[V5_ATTEMPT_RECEIPT_REL],
            "v5_blocked_digest": V5_SEALED_DIGESTS[V5_BLOCKED_REL],
            "v1_v2_v3_v4_v5_preservation": (
                "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
            ),
        },
        "accepted_v5_engineering_history": {
            "http_200_parse_failure_observability": (
                "ALLOWLISTED_CONTENT_FREE_REASON_CODE"
            ),
            "http_error_persistence": "STATUS_ONLY_NO_PROVIDER_BODY",
            "sqlite_schema_changed": False,
            "accepted_v5_artifacts_rewritten": False,
        },
        "authorized_v6_protocol_change": {
            "only_scientific_protocol_change": (
                "EXACT_REQUEST_ALIAS_TO_EXACT_RETURNED_SNAPSHOT_BINDING"
            ),
            "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
            "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
            "matching": "TWO_EXACT_LITERAL_EQUALITIES_FAIL_CLOSED",
            "generic_alias_acceptance": "FORBIDDEN",
        },
        "provider_release_contract": release,
        "exact_provider_contract": exact,
        "response_contract_equivalence": deepcopy(
            v5["response_contract_equivalence"]
        ),
        "bounded_retry": {
            "policy_ref": _repo_ref(V6_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "maximum_physical_attempts": 3,
            "maximum_retry_count": 2,
            "deterministic_backoff_ms": [1000, 3000],
            "physical_attempt_identities": physical_identities,
            "sqlite_calls_schema_digest": v5["bounded_retry"][
                "sqlite_calls_schema_digest"
            ],
        },
        "preserved_scientific_identity": _v6_preserved_scientific_identity(
            repo_root
        ),
        "future_r1_scientific_contract": future,
        "inherited_v5_scientific_contract_digest": sha256_digest(
            v5["future_r1_scientific_contract"]
        ),
        "v6_scientific_contract_digest": sha256_digest(future),
        "pre_outcome_counters": deepcopy(v5["pre_outcome_counters"]),
        "r1_worker_launch_authorized": False,
    }


def expected_v6_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v6_manifest(repo_root)
    policy = expected_v6_retry_policy(repo_root)
    release = expected_v6_provider_release(repo_root)
    return {
        "schema": V6_AUTH_SCHEMA,
        "status": "AUTHORIZED_ONE_V6_EXACT_MODEL_PAIR_DIAGNOSTIC_SLOT",
        "attempt_id": V6_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V6_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V6_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "provider_release_ref": _repo_ref(V6_RELEASE_REL),
        "provider_release_artifact_digest": bytes_sha256(
            canonical_json_bytes(release)
        ),
        "provider_release_contract_digest": release["release_digest"],
        "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
        "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v6(repo_root: Path) -> dict[str, Any]:
    verify_v5_seal(repo_root)
    release = _load_exact(
        repo_root / V6_RELEASE_REL, expected_v6_provider_release(repo_root)
    )
    policy = _load_exact(
        repo_root / V6_POLICY_REL, expected_v6_retry_policy(repo_root)
    )
    manifest = _load_exact(
        repo_root / V6_MANIFEST_REL, expected_prefreeze_v6_manifest(repo_root)
    )
    _load_exact(repo_root / V6_AUTH_REL, expected_v6_authorization(repo_root))
    exact = manifest["exact_provider_contract"]
    validate_v6_exact_model_pair(
        requested_model_alias=exact["requested_model_alias"],
        returned_model=exact["required_returned_snapshot"],
    )
    if (
        exact["provider_release_digest"] != release["release_digest"]
        or exact["request_payload_digest"]
        != _read_json(repo_root / V5_MANIFEST_REL)["exact_provider_contract"][
            "request_payload_digest"
        ]
    ):
        raise Wave2IntegrationError("V6 release or request payload binding changed")
    future = manifest["future_r1_scientific_contract"]
    shared = future["shared_call_contract_digest"]
    if (
        future["side_a_call_contract_digest"] != shared
        or future["side_b_call_contract_digest"] != shared
        or future["proposal_slots_per_side"] != 8
        or future["proposal_denominator_per_side"] != 8
    ):
        raise Wave2IntegrationError("V6 A/B symmetry or denominator changed")
    if (
        policy["diagnostic_slot"]["maximum_total_physical_attempts"] != 3
        or policy["diagnostic_slot"]["deterministic_backoff_ms_after_failure"]
        != [1000, 3000]
        or policy["exact_model_pair_failure"]["retry_eligible"] is not False
    ):
        raise Wave2IntegrationError("V6 retry or model-pair failure policy changed")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V6 is not pre-outcome")
    return manifest


def provider_free_v6_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v6(repo_root)
    return {
        "schema": V6_DRY_RUN_SCHEMA,
        "status": "PASS_PROVIDER_FREE_V6_EXACT_MODEL_PAIR",
        "attempt_id": V6_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_v3_v4_v5_seals_verified": True,
        "exact_requested_alias_verified": True,
        "exact_required_returned_snapshot_verified": True,
        "pair_bound_provider_release_verified": True,
        "exact_v5_request_payload_verified": True,
        "local_uniqueness_before_downstream_verified": True,
        "ab_call_contract_symmetry_verified": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def v7_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V7 physical attempt ordinal must be 1..3")
    return V7_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v6_seal(repo_root: Path) -> dict[str, str]:
    verify_v5_seal(repo_root)
    for relative, digest in V6_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V6 bytes changed: {relative.as_posix()}"
            )
    return {
        relative.as_posix(): digest
        for relative, digest in V6_SEALED_DIGESTS.items()
    }


def exact_v7_probe_request_payload_digest(repo_root: Path) -> str:
    payload = exact_v4_probe_request_payload(repo_root)
    payload["max_tokens"] = V7_DIAGNOSTIC_TOKEN_CEILING
    return sha256_digest(payload)


def expected_v7_provider_release(repo_root: Path) -> dict[str, Any]:
    verify_v6_seal(repo_root)
    release = deepcopy(_read_json(repo_root / V6_RELEASE_REL))
    release.pop("release_digest")
    release["schema"] = "recclaw.research-line.provider-release-contract.v7"
    release["diagnostic_token_budget"] = V7_DIAGNOSTIC_TOKEN_CEILING
    return {**release, "release_digest": sha256_digest(release)}


def expected_v7_retry_policy(repo_root: Path) -> dict[str, Any]:
    verify_v6_seal(repo_root)
    policy = deepcopy(_read_json(repo_root / V6_POLICY_REL))
    policy["schema"] = "recclaw.research-line.r1-provider-retry-policy.v7"
    policy.pop("inherited_v5_policy_ref", None)
    policy.pop("inherited_v5_policy_digest", None)
    policy["inherited_v6_policy_ref"] = _repo_ref(V6_POLICY_REL)
    policy["inherited_v6_policy_digest"] = V6_SEALED_DIGESTS[V6_POLICY_REL]
    policy["diagnostic_slot"]["slot_id"] = (
        "PREFREEZE_V7_DIAGNOSTIC_CEILING_ALIGNMENT"
    )
    return policy


def expected_prefreeze_v7_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v6_seal(repo_root)
    v6 = _read_json(repo_root / V6_MANIFEST_REL)
    release = expected_v7_provider_release(repo_root)
    policy = expected_v7_retry_policy(repo_root)
    manifest = deepcopy(v6)
    manifest["schema"] = "recclaw.research-line.r1-r2-prefreeze-attempt.v7"
    manifest["attempt_identity"] = {
        "attempt_id": V7_ATTEMPT_ID,
        "base_commit": V6_HEAD,
        "base_parent": V6_PARENT,
        "base_tree": V6_TREE,
        "pre_outcome": True,
        "distinct_from_v1_v2_v3_v4_v5_v6_attempts": True,
        "old_attempt_call_session_db_identity_reuse": False,
    }
    manifest["sealed_predecessor_evidence"] = {
        "v6_manifest_digest": V6_SEALED_DIGESTS[V6_MANIFEST_REL],
        "v6_attempt_digest": V6_SEALED_DIGESTS[V6_ATTEMPT_RECEIPT_REL],
        "v6_blocked_digest": V6_SEALED_DIGESTS[V6_BLOCKED_REL],
        "v1_v2_v3_v4_v5_v6_preservation": (
            "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
        ),
    }
    manifest["authorized_v7_infrastructure_alignment"] = {
        "decision": "CONTRACT_EQUIVALENT_INFRASTRUCTURE_FIX",
        "only_contract_delta": (
            "DIAGNOSTIC_TOKEN_CEILING_2000_TO_FROZEN_6000"
        ),
        "prior_diagnostic_token_ceiling": 2000,
        "diagnostic_token_ceiling": V7_DIAGNOSTIC_TOKEN_CEILING,
        "transport_and_future_r1_token_ceiling": 6000,
        "research_proposal_budget_changed": False,
        "schema_semantics_model_pair_or_uniqueness_changed": False,
    }
    manifest["provider_release_contract"] = release
    exact = manifest["exact_provider_contract"]
    exact.update(
        {
            "provider_release_ref": _repo_ref(V7_RELEASE_REL),
            "provider_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(release)
            ),
            "provider_release_digest": release["release_digest"],
            "request_payload_digest": exact_v7_probe_request_payload_digest(
                repo_root
            ),
            "token_budget": V7_DIAGNOSTIC_TOKEN_CEILING,
            "logical_call_id": V7_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V7_SESSION_ID,
            "diagnostic_slot_id": (
                "PREFREEZE_V7_DIAGNOSTIC_CEILING_ALIGNMENT"
            ),
        }
    )
    identities = [
        {
            "ordinal": ordinal,
            "physical_attempt_identity_digest": sha256_digest(
                {
                    "attempt_id": V7_ATTEMPT_ID,
                    "diagnostic_slot": (
                        "PREFREEZE_V7_DIAGNOSTIC_CEILING_ALIGNMENT"
                    ),
                    "ordinal": ordinal,
                    "private_root_digest": sha256_digest(
                        {"path": v7_physical_root(ordinal).as_posix()}
                    ),
                }
            ),
            "private_root_digest": sha256_digest(
                {"path": v7_physical_root(ordinal).as_posix()}
            ),
        }
        for ordinal in (1, 2, 3)
    ]
    manifest["bounded_retry"].update(
        {
            "policy_ref": _repo_ref(V7_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "physical_attempt_identities": identities,
        }
    )
    future = manifest["future_r1_scientific_contract"]
    manifest.pop("inherited_v5_scientific_contract_digest", None)
    manifest.pop("v6_scientific_contract_digest", None)
    manifest["inherited_v6_scientific_contract_digest"] = sha256_digest(
        v6["future_r1_scientific_contract"]
    )
    manifest["v7_scientific_contract_digest"] = sha256_digest(future)
    return manifest


def expected_v7_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v7_manifest(repo_root)
    policy = expected_v7_retry_policy(repo_root)
    release = expected_v7_provider_release(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v7-authorization.v1",
        "status": "AUTHORIZED_ONE_V7_CEILING_ALIGNMENT_DIAGNOSTIC_SLOT",
        "attempt_id": V7_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V7_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V7_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "provider_release_ref": _repo_ref(V7_RELEASE_REL),
        "provider_release_artifact_digest": bytes_sha256(
            canonical_json_bytes(release)
        ),
        "provider_release_contract_digest": release["release_digest"],
        "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
        "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
        "diagnostic_token_ceiling": V7_DIAGNOSTIC_TOKEN_CEILING,
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v7(repo_root: Path) -> dict[str, Any]:
    verify_v6_seal(repo_root)
    release = _load_exact(
        repo_root / V7_RELEASE_REL, expected_v7_provider_release(repo_root)
    )
    policy = _load_exact(
        repo_root / V7_POLICY_REL, expected_v7_retry_policy(repo_root)
    )
    manifest = _load_exact(
        repo_root / V7_MANIFEST_REL, expected_prefreeze_v7_manifest(repo_root)
    )
    _load_exact(repo_root / V7_AUTH_REL, expected_v7_authorization(repo_root))
    v6 = _read_json(repo_root / V6_MANIFEST_REL)
    exact = manifest["exact_provider_contract"]
    validate_v6_exact_model_pair(
        requested_model_alias=exact["requested_model_alias"],
        returned_model=exact["required_returned_snapshot"],
    )
    allowed_exact_delta = {
        "provider_release_ref",
        "provider_release_artifact_digest",
        "provider_release_digest",
        "request_payload_digest",
        "token_budget",
        "logical_call_id",
        "proposal_generation_session_id",
        "diagnostic_slot_id",
    }
    if {
        key: value
        for key, value in exact.items()
        if key not in allowed_exact_delta
    } != {
        key: value
        for key, value in v6["exact_provider_contract"].items()
        if key not in allowed_exact_delta
    }:
        raise Wave2IntegrationError("V7 changed a non-diagnostic Provider field")
    if (
        exact["token_budget"] != V7_DIAGNOSTIC_TOKEN_CEILING
        or release["diagnostic_token_budget"]
        != release["transport_max_total_tokens_per_call"]
        or release["diagnostic_token_budget"]
        != release["future_r1_token_budget_per_call"]
    ):
        raise Wave2IntegrationError("V7 diagnostic ceiling is not exact 6000")
    if (
        manifest["future_r1_scientific_contract"]
        != v6["future_r1_scientific_contract"]
        or manifest["response_contract_equivalence"]
        != v6["response_contract_equivalence"]
        or manifest["preserved_scientific_identity"]
        != v6["preserved_scientific_identity"]
    ):
        raise Wave2IntegrationError("V7 changed the R1 or response contract")
    if (
        policy["diagnostic_slot"]["maximum_total_physical_attempts"] != 3
        or policy["diagnostic_slot"]["deterministic_backoff_ms_after_failure"]
        != [1000, 3000]
        or policy["response_contract_failure"]["retry_eligible"] is not False
        or "SEMANTIC_RESPONSE_CONTRACT_FAILURE"
        not in policy["diagnostic_slot"]["terminal_no_retry_failure_classes"]
    ):
        raise Wave2IntegrationError("V7 retry/terminal policy changed")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V7 is not pre-outcome")
    return manifest


def provider_free_v7_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v7(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v7-dry-run-receipt.v1",
        "status": "PASS_PROVIDER_FREE_V7_DIAGNOSTIC_CEILING_ALIGNMENT",
        "attempt_id": V7_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_v3_v4_v5_v6_seals_verified": True,
        "only_diagnostic_ceiling_delta_verified": True,
        "diagnostic_transport_future_ceiling_equal_6000": True,
        "future_r1_contract_exact_v6_verified": True,
        "exact_model_pair_verified": True,
        "strict_schema_sentinel_local_uniqueness_verified": True,
        "token_ceiling_terminal_verified": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def v8_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V8 physical attempt ordinal must be 1..3")
    return V8_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v7_seal(repo_root: Path) -> dict[str, str]:
    """Verify accepted V1--V7 artifacts without freezing mutable runtime code."""

    verified = verify_v4_seal(repo_root)
    for sealed in (V5_SEALED_DIGESTS, V6_SEALED_DIGESTS, V7_SEALED_DIGESTS):
        for relative, digest in sealed.items():
            path = repo_root / relative
            if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
                raise Wave2IntegrationError(
                    f"sealed predecessor bytes changed: {relative.as_posix()}"
                )
            verified[relative.as_posix()] = digest
    return verified


def exact_v8_probe_request_payload(repo_root: Path) -> dict[str, Any]:
    payload = exact_v4_probe_request_payload(repo_root)
    payload["model"] = V8_MODEL_SNAPSHOT
    payload["max_tokens"] = V8_DIAGNOSTIC_TOKEN_CEILING
    return payload


def exact_v8_probe_request_payload_digest(repo_root: Path) -> str:
    return sha256_digest(exact_v8_probe_request_payload(repo_root))


def validate_v8_exact_snapshot_pair(
    *, requested_model: str, returned_model: str
) -> None:
    if requested_model != V8_MODEL_SNAPSHOT:
        raise Wave2IntegrationError("V8 requested snapshot mismatch")
    if returned_model != V8_MODEL_SNAPSHOT:
        raise Wave2IntegrationError("V8 returned snapshot mismatch")


def expected_v8_broker_release(repo_root: Path) -> dict[str, Any]:
    v7 = _read_json(repo_root / V7_MANIFEST_REL)
    exact = v7["exact_provider_contract"]
    payload = {
        "endpoint_digest": exact["endpoint_digest"],
        "max_total_tokens_per_call": V8_DIAGNOSTIC_TOKEN_CEILING,
        "model": V8_MODEL_SNAPSHOT,
        "request_mode": "SINGLE_JSON_SCHEMA_NO_TOOLS",
        "response_schema_digest": exact["response_schema_digest"],
        "retry_count": 0,
        "temperature": 0.0,
        "timeout_ms": 900_000,
        "transport": "HTTPS_CHAT_COMPLETIONS_V1",
    }
    release = LabApiBrokerReleaseV1(
        **payload, release_digest=sha256_digest(payload)
    )
    release.verify()
    return release.to_dict()


_MODEL_IDENTITY_BINDING_FIELDS = (
    "requested_model_literal",
    "required_returned_snapshot",
    "endpoint_digest",
    "credential_config_digest",
    "credential_identity_digest",
    "response_schema_digest",
    "local_uniqueness_contract_digest",
    "prompt_digest",
    "tool_policy_digest",
    "request_mode",
    "temperature",
    "diagnostic_token_budget",
    "transport_max_total_tokens_per_call",
    "future_r1_token_budget_per_call",
    "future_r1_call_count_per_side",
    "future_r1_proposal_budget_per_side",
)


def _model_identity_contract_digest(release: Mapping[str, Any]) -> str:
    return sha256_digest(
        {key: release[key] for key in _MODEL_IDENTITY_BINDING_FIELDS}
    )


def expected_v8_provider_release(repo_root: Path) -> dict[str, Any]:
    verify_v7_seal(repo_root)
    release = deepcopy(_read_json(repo_root / V7_RELEASE_REL))
    release.pop("release_digest")
    release.pop("requested_model_alias")
    broker_release = expected_v8_broker_release(repo_root)
    release.update(
        {
            "schema": "recclaw.research-line.provider-release-contract.v8",
            "requested_model_literal": V8_MODEL_SNAPSHOT,
            "required_returned_snapshot": V8_MODEL_SNAPSHOT,
            "request_and_return_matching": "TWO_EXACT_LITERAL_EQUALITIES",
            "alias_request_or_fallback": "FORBIDDEN",
            "transport_release_ref": "INLINE_LAB_API_BROKER_RELEASE_V1_V8",
            "transport_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(broker_release)
            ),
            "transport_release_contract_digest": broker_release[
                "release_digest"
            ],
            "broker_source_digest": bytes_sha256(
                (repo_root / BROKER_SOURCE_REL).read_bytes()
            ),
        }
    )
    release["model_identity_contract_digest"] = (
        _model_identity_contract_digest(release)
    )
    return {**release, "release_digest": sha256_digest(release)}


def expected_v8_retry_policy(repo_root: Path) -> dict[str, Any]:
    verify_v7_seal(repo_root)
    policy = deepcopy(_read_json(repo_root / V7_POLICY_REL))
    policy["schema"] = "recclaw.research-line.r1-provider-retry-policy.v8"
    policy["inherited_v7_policy_ref"] = _repo_ref(V7_POLICY_REL)
    policy["inherited_v7_policy_digest"] = V7_SEALED_DIGESTS[V7_POLICY_REL]
    policy.pop("inherited_v6_policy_ref", None)
    policy.pop("inherited_v6_policy_digest", None)
    policy["diagnostic_slot"]["slot_id"] = (
        "PREFREEZE_V8_EXACT_SNAPSHOT_REQUEST"
    )
    policy["response_contract_reason_vocabulary"] = (
        diagnostic_reason_vocabulary(include_returned_model_identity=True)
    )
    policy.pop("exact_model_pair_failure", None)
    policy["exact_snapshot_identity_failure"] = {
        "failure_class": "RETURNED_MODEL_SNAPSHOT_MISMATCH",
        "retry_eligible": False,
        "manual_patch": "FORBIDDEN",
        "successful_response_selection": "FORBIDDEN",
        "candidate_admission": "FORBIDDEN",
        "mechanism_negative_evidence": False,
    }
    return policy


def expected_prefreeze_v8_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v7_seal(repo_root)
    v7 = _read_json(repo_root / V7_MANIFEST_REL)
    release = expected_v8_provider_release(repo_root)
    policy = expected_v8_retry_policy(repo_root)
    manifest = deepcopy(v7)
    manifest["schema"] = "recclaw.research-line.r1-r2-prefreeze-attempt.v8"
    manifest["attempt_identity"] = {
        "attempt_id": V8_ATTEMPT_ID,
        "base_commit": V7_HEAD,
        "base_parent": V7_PARENT,
        "base_tree": V7_TREE,
        "pre_outcome": True,
        "distinct_from_v1_v2_v3_v4_v5_v6_v7_attempts": True,
        "old_attempt_call_session_db_identity_reuse": False,
    }
    manifest["sealed_predecessor_evidence"] = {
        "v7_manifest_digest": V7_SEALED_DIGESTS[V7_MANIFEST_REL],
        "v7_attempt_digest": V7_SEALED_DIGESTS[V7_ATTEMPT_RECEIPT_REL],
        "v7_blocked_digest": V7_SEALED_DIGESTS[V7_BLOCKED_REL],
        "v1_v2_v3_v4_v5_v6_v7_preservation": (
            "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
        ),
    }
    manifest.pop("authorized_v7_infrastructure_alignment", None)
    manifest["authorized_v8_protocol_change"] = {
        "only_scientific_contract_delta": (
            "EXACT_SNAPSHOT_REQUEST_AND_EXACT_SAME_SNAPSHOT_RETURN"
        ),
        "requested_model_literal": V8_MODEL_SNAPSHOT,
        "required_returned_snapshot": V8_MODEL_SNAPSHOT,
        "alias_allowlist_prefix_regex_startswith_or_fallback": "FORBIDDEN",
        "endpoint_schema_prompt_sentinel_tools_temperature_budgets_unchanged": True,
    }
    manifest["provider_returned_model_evidence"] = {
        "source": "PROVIDER_ENVELOPE_MODEL_FIELD_ONLY",
        "missing_non_string_or_empty_reason": (
            "RETURNED_MODEL_TYPE_OR_EMPTY"
        ),
        "missing_non_string_or_empty_retry_eligible": False,
        "request_model_fallback_for_identity_proof": "FORBIDDEN",
        "broker_source_ref": _repo_ref(BROKER_SOURCE_REL),
        "broker_source_digest": release["broker_source_digest"],
    }
    manifest["provider_release_contract"] = release
    exact = manifest["exact_provider_contract"]
    exact.pop("requested_model_alias", None)
    exact.update(
        {
            "model": V8_MODEL_SNAPSHOT,
            "model_digest": sha256_digest({"model": V8_MODEL_SNAPSHOT}),
            "requested_model_literal": V8_MODEL_SNAPSHOT,
            "required_returned_snapshot": V8_MODEL_SNAPSHOT,
            "model_identity_pair_digest": sha256_digest(
                {
                    "requested_model_literal": V8_MODEL_SNAPSHOT,
                    "required_returned_snapshot": V8_MODEL_SNAPSHOT,
                }
            ),
            "transport_provider_release_digest": release[
                "transport_release_contract_digest"
            ],
            "provider_release_ref": _repo_ref(V8_RELEASE_REL),
            "provider_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(release)
            ),
            "provider_release_digest": release["release_digest"],
            "request_payload_digest": exact_v8_probe_request_payload_digest(
                repo_root
            ),
            "token_budget": V8_DIAGNOSTIC_TOKEN_CEILING,
            "logical_call_id": V8_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V8_SESSION_ID,
            "diagnostic_slot_id": "PREFREEZE_V8_EXACT_SNAPSHOT_REQUEST",
        }
    )
    identities = []
    for ordinal in (1, 2, 3):
        private_digest = sha256_digest(
            {"path": v8_physical_root(ordinal).as_posix()}
        )
        identities.append(
            {
                "ordinal": ordinal,
                "physical_attempt_identity_digest": sha256_digest(
                    {
                        "attempt_id": V8_ATTEMPT_ID,
                        "diagnostic_slot": (
                            "PREFREEZE_V8_EXACT_SNAPSHOT_REQUEST"
                        ),
                        "ordinal": ordinal,
                        "private_root_digest": private_digest,
                    }
                ),
                "private_root_digest": private_digest,
            }
        )
    manifest["bounded_retry"].update(
        {
            "policy_ref": _repo_ref(V8_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "physical_attempt_identities": identities,
        }
    )
    future = manifest["future_r1_scientific_contract"]
    inherited_shared = future["shared_call_contract_digest"]
    shared = sha256_digest(
        {
            "inherited_v7_shared_call_contract_digest": inherited_shared,
            "v8_provider_release_contract_digest": release["release_digest"],
        }
    )
    future.update(
        {
            "requested_model_literal": V8_MODEL_SNAPSHOT,
            "required_returned_snapshot": V8_MODEL_SNAPSHOT,
            "shared_call_contract_digest": shared,
            "side_a_call_contract_digest": shared,
            "side_b_call_contract_digest": shared,
        }
    )
    manifest.pop("inherited_v6_scientific_contract_digest", None)
    manifest.pop("v7_scientific_contract_digest", None)
    manifest["inherited_v7_scientific_contract_digest"] = sha256_digest(
        v7["future_r1_scientific_contract"]
    )
    manifest["v8_scientific_contract_digest"] = sha256_digest(future)
    manifest["r1_worker_launch_authorized"] = False
    return manifest


def expected_v8_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v8_manifest(repo_root)
    policy = expected_v8_retry_policy(repo_root)
    release = expected_v8_provider_release(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v8-authorization.v1",
        "status": "AUTHORIZED_ONE_V8_EXACT_SNAPSHOT_DIAGNOSTIC_SLOT",
        "attempt_id": V8_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V8_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V8_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "provider_release_ref": _repo_ref(V8_RELEASE_REL),
        "provider_release_artifact_digest": bytes_sha256(
            canonical_json_bytes(release)
        ),
        "provider_release_contract_digest": release["release_digest"],
        "broker_source_digest": release["broker_source_digest"],
        "requested_model_literal": V8_MODEL_SNAPSHOT,
        "required_returned_snapshot": V8_MODEL_SNAPSHOT,
        "diagnostic_token_ceiling": V8_DIAGNOSTIC_TOKEN_CEILING,
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v8(repo_root: Path) -> dict[str, Any]:
    verify_v7_seal(repo_root)
    release = _load_exact(
        repo_root / V8_RELEASE_REL, expected_v8_provider_release(repo_root)
    )
    policy = _load_exact(
        repo_root / V8_POLICY_REL, expected_v8_retry_policy(repo_root)
    )
    manifest = _load_exact(
        repo_root / V8_MANIFEST_REL, expected_prefreeze_v8_manifest(repo_root)
    )
    _load_exact(repo_root / V8_AUTH_REL, expected_v8_authorization(repo_root))
    exact = manifest["exact_provider_contract"]
    validate_v8_exact_snapshot_pair(
        requested_model=exact["requested_model_literal"],
        returned_model=exact["required_returned_snapshot"],
    )
    if (
        exact["request_payload_digest"]
        != exact_v8_probe_request_payload_digest(repo_root)
        or exact["provider_release_digest"] != release["release_digest"]
        or release["diagnostic_token_budget"] != 6000
        or release["transport_max_total_tokens_per_call"] != 6000
        or release["future_r1_token_budget_per_call"] != 6000
    ):
        raise Wave2IntegrationError("V8 exact payload/release/budget changed")
    v7 = _read_json(repo_root / V7_MANIFEST_REL)
    allowed_exact_delta = {
        "model",
        "model_digest",
        "requested_model_alias",
        "requested_model_literal",
        "required_returned_snapshot",
        "model_identity_pair_digest",
        "transport_provider_release_digest",
        "provider_release_ref",
        "provider_release_artifact_digest",
        "provider_release_digest",
        "request_payload_digest",
        "logical_call_id",
        "proposal_generation_session_id",
        "diagnostic_slot_id",
    }
    if {
        key: value for key, value in exact.items() if key not in allowed_exact_delta
    } != {
        key: value
        for key, value in v7["exact_provider_contract"].items()
        if key not in allowed_exact_delta
    }:
        raise Wave2IntegrationError("V8 changed an unauthorized Provider field")
    v7_future = deepcopy(v7["future_r1_scientific_contract"])
    v8_future = deepcopy(manifest["future_r1_scientific_contract"])
    for field in (
        "requested_model_literal",
        "required_returned_snapshot",
        "shared_call_contract_digest",
        "side_a_call_contract_digest",
        "side_b_call_contract_digest",
    ):
        v7_future.pop(field, None)
        v8_future.pop(field, None)
    if (
        v8_future != v7_future
        or manifest["response_contract_equivalence"]
        != v7["response_contract_equivalence"]
        or manifest["preserved_scientific_identity"]
        != v7["preserved_scientific_identity"]
    ):
        raise Wave2IntegrationError("V8 changed the preserved R1 contract")
    future = manifest["future_r1_scientific_contract"]
    if (
        future["proposal_slots_per_side"] != 8
        or future["proposal_denominator_per_side"] != 8
        or future["side_a_call_contract_digest"]
        != future["shared_call_contract_digest"]
        or future["side_b_call_contract_digest"]
        != future["shared_call_contract_digest"]
    ):
        raise Wave2IntegrationError("V8 A/B symmetry or denominator changed")
    if (
        policy["response_contract_reason_vocabulary"]
        != diagnostic_reason_vocabulary(include_returned_model_identity=True)
        or policy["exact_snapshot_identity_failure"]["retry_eligible"] is not False
        or policy["diagnostic_slot"]["maximum_total_physical_attempts"] != 3
        or policy["diagnostic_slot"]["deterministic_backoff_ms_after_failure"]
        != [1000, 3000]
    ):
        raise Wave2IntegrationError("V8 terminal/retry policy changed")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V8 is not pre-outcome")
    return manifest


def provider_free_v8_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v8(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v8-dry-run-receipt.v1",
        "status": "PASS_PROVIDER_FREE_V8_EXACT_SNAPSHOT_REQUEST",
        "attempt_id": V8_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_v3_v4_v5_v6_v7_seals_verified": True,
        "exact_snapshot_request_and_return_verified": True,
        "provider_envelope_model_required_without_fallback": True,
        "endpoint_schema_prompt_sentinel_tools_temperature_verified_unchanged": True,
        "diagnostic_transport_future_ceiling_equal_6000": True,
        "future_r1_ab_slots_denominator_seed_threshold_analysis_verified": True,
        "local_uniqueness_before_projection_and_downstream_verified": True,
        "response_contract_and_identity_failures_terminal_verified": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def prefreeze_v8_runtime_spec() -> dict[str, Any]:
    """One shared V8 delta consumed by builder, probe, and finalizer."""

    return {
        "label": "V8",
        "attempt_id": V8_ATTEMPT_ID,
        "attempt_rel": V8_ATTEMPT_RECEIPT_REL,
        "attempt_schema": V8_ATTEMPT_RECEIPT_SCHEMA,
        "auth_rel": V8_AUTH_REL,
        "blocked_rel": V8_BLOCKED_REL,
        "blocked_schema": V8_BLOCKED_SCHEMA,
        "diagnostic_token_ceiling": V8_DIAGNOSTIC_TOKEN_CEILING,
        "token_ceiling": V8_DIAGNOSTIC_TOKEN_CEILING,
        "dry_run_rel": V8_DRY_RUN_REL,
        "manifest_rel": V8_MANIFEST_REL,
        "policy_rel": V8_POLICY_REL,
        "private_root": V8_PRIVATE_ROOT,
        "ready_rel": V8_READY_REL,
        "ready_schema": V8_READY_SCHEMA,
        "release_rel": V8_RELEASE_REL,
        "verification_rel": V8_VERIFICATION_REL,
        "verification_schema": V8_VERIFICATION_SCHEMA,
        "validate": validate_prefreeze_v8,
        "dry_run": provider_free_v8_dry_run,
        "requested_model": V8_MODEL_SNAPSHOT,
        "required_returned_model": V8_MODEL_SNAPSHOT,
        "validate_model_pair": validate_v8_exact_snapshot_pair,
        "identity_receipt_fields": {
            "requested_model_literal": V8_MODEL_SNAPSHOT,
            "required_returned_snapshot": V8_MODEL_SNAPSHOT,
        },
        "blocked_identity_field": "exact_snapshot_request_and_return",
        "pass_classification": (
            "PASS_EXACT_GPT_5_4_2026_03_05_REQUEST_RETURN_AUTH_SCHEMA"
        ),
        "pass_status": "PASS_PROVIDER_FREE_V8_EXACT_SNAPSHOT_REQUEST",
        "verify_predecessor": verify_v7_seal,
        "expected_policy": expected_v8_retry_policy,
        "expected_manifest": expected_prefreeze_v8_manifest,
        "expected_authorization": expected_v8_authorization,
        "expected_release": expected_v8_provider_release,
        "logical_call_id": V8_LOGICAL_CALL_ID,
        "session_id": V8_SESSION_ID,
        "slot_id": "PREFREEZE_V8_EXACT_SNAPSHOT_REQUEST",
        "physical_root": v8_physical_root,
    }


def v9_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V9 physical attempt ordinal must be 1..3")
    return V9_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v8_seal(repo_root: Path) -> dict[str, str]:
    verified = verify_v7_seal(repo_root)
    for relative, digest in V8_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V8 bytes changed: {relative.as_posix()}"
            )
        verified[relative.as_posix()] = digest
    return verified


def expected_v9_provider_release(repo_root: Path) -> dict[str, Any]:
    verify_v8_seal(repo_root)
    release = deepcopy(_read_json(repo_root / V8_RELEASE_REL))
    release.pop("release_digest")
    release["schema"] = "recclaw.research-line.provider-release-contract.v9"
    release["availability_predecessor"] = {
        "v8_release_digest": V8_SEALED_DIGESTS[V8_RELEASE_REL],
        "v8_attempt_digest": V8_SEALED_DIGESTS[V8_ATTEMPT_RECEIPT_REL],
        "v8_blocked_digest": V8_SEALED_DIGESTS[V8_BLOCKED_REL],
        "v8_terminal_classification": "HTTP_503_TRANSIENT_PROVIDER_ERROR",
        "v8_physical_calls": 3,
        "v8_retry_count": 2,
    }
    return {**release, "release_digest": sha256_digest(release)}


def expected_v9_retry_policy(repo_root: Path) -> dict[str, Any]:
    verify_v8_seal(repo_root)
    policy = deepcopy(_read_json(repo_root / V8_POLICY_REL))
    policy["schema"] = "recclaw.research-line.r1-provider-retry-policy.v9"
    policy.pop("inherited_v7_policy_ref", None)
    policy.pop("inherited_v7_policy_digest", None)
    policy["inherited_v8_policy_ref"] = _repo_ref(V8_POLICY_REL)
    policy["inherited_v8_policy_digest"] = V8_SEALED_DIGESTS[V8_POLICY_REL]
    policy["diagnostic_slot"]["slot_id"] = (
        "PREFREEZE_V9_AVAILABILITY_RECHECK"
    )
    return policy


def expected_prefreeze_v9_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v8_seal(repo_root)
    v8 = _read_json(repo_root / V8_MANIFEST_REL)
    release = expected_v9_provider_release(repo_root)
    policy = expected_v9_retry_policy(repo_root)
    manifest = deepcopy(v8)
    manifest["schema"] = "recclaw.research-line.r1-r2-prefreeze-attempt.v9"
    manifest["attempt_identity"] = {
        "attempt_id": V9_ATTEMPT_ID,
        "base_commit": V8_HEAD,
        "base_parent": V8_PARENT,
        "base_tree": V8_TREE,
        "pre_outcome": True,
        "distinct_from_v1_v2_v3_v4_v5_v6_v7_v8_attempts": True,
        "old_attempt_call_session_db_identity_reuse": False,
    }
    manifest["sealed_predecessor_evidence"] = {
        "v8_manifest_digest": V8_SEALED_DIGESTS[V8_MANIFEST_REL],
        "v8_attempt_digest": V8_SEALED_DIGESTS[V8_ATTEMPT_RECEIPT_REL],
        "v8_blocked_digest": V8_SEALED_DIGESTS[V8_BLOCKED_REL],
        "v1_v2_v3_v4_v5_v6_v7_v8_preservation": (
            "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
        ),
    }
    manifest["authorized_v9_availability_recheck"] = {
        "scope": "ONE_FRESH_BOUNDED_DIAGNOSTIC_SLOT_ONLY",
        "predecessor": "V8_THREE_HTTP_503_TRANSIENT_ATTEMPTS_EXHAUSTED",
        "scientific_contract_change": False,
        "request_payload_change": False,
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "v10_or_unbounded_retry_authorized": False,
    }
    manifest["provider_release_contract"] = release
    exact = manifest["exact_provider_contract"]
    exact.update(
        {
            "provider_release_ref": _repo_ref(V9_RELEASE_REL),
            "provider_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(release)
            ),
            "provider_release_digest": release["release_digest"],
            "logical_call_id": V9_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V9_SESSION_ID,
            "diagnostic_slot_id": "PREFREEZE_V9_AVAILABILITY_RECHECK",
        }
    )
    identities = []
    for ordinal in (1, 2, 3):
        private_digest = sha256_digest(
            {"path": v9_physical_root(ordinal).as_posix()}
        )
        identities.append(
            {
                "ordinal": ordinal,
                "physical_attempt_identity_digest": sha256_digest(
                    {
                        "attempt_id": V9_ATTEMPT_ID,
                        "diagnostic_slot": (
                            "PREFREEZE_V9_AVAILABILITY_RECHECK"
                        ),
                        "ordinal": ordinal,
                        "private_root_digest": private_digest,
                    }
                ),
                "private_root_digest": private_digest,
            }
        )
    manifest["bounded_retry"].update(
        {
            "policy_ref": _repo_ref(V9_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "physical_attempt_identities": identities,
        }
    )
    v8_scientific = sha256_digest(v8["future_r1_scientific_contract"])
    manifest["inherited_v8_scientific_contract_digest"] = v8_scientific
    manifest["v9_scientific_contract_digest"] = v8_scientific
    manifest["r1_worker_launch_authorized"] = False
    return manifest


def expected_v9_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v9_manifest(repo_root)
    policy = expected_v9_retry_policy(repo_root)
    release = expected_v9_provider_release(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v9-authorization.v1",
        "status": "AUTHORIZED_ONE_V9_AVAILABILITY_RECHECK_SLOT",
        "attempt_id": V9_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V9_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V9_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "provider_release_ref": _repo_ref(V9_RELEASE_REL),
        "provider_release_artifact_digest": bytes_sha256(
            canonical_json_bytes(release)
        ),
        "provider_release_contract_digest": release["release_digest"],
        "requested_model_literal": V8_MODEL_SNAPSHOT,
        "required_returned_snapshot": V8_MODEL_SNAPSHOT,
        "diagnostic_token_ceiling": V8_DIAGNOSTIC_TOKEN_CEILING,
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v9(repo_root: Path) -> dict[str, Any]:
    verify_v8_seal(repo_root)
    release = _load_exact(
        repo_root / V9_RELEASE_REL, expected_v9_provider_release(repo_root)
    )
    policy = _load_exact(
        repo_root / V9_POLICY_REL, expected_v9_retry_policy(repo_root)
    )
    manifest = _load_exact(
        repo_root / V9_MANIFEST_REL, expected_prefreeze_v9_manifest(repo_root)
    )
    _load_exact(repo_root / V9_AUTH_REL, expected_v9_authorization(repo_root))
    v8 = _read_json(repo_root / V8_MANIFEST_REL)
    exact8 = v8["exact_provider_contract"]
    exact9 = manifest["exact_provider_contract"]
    allowed_exact_delta = {
        "provider_release_ref",
        "provider_release_artifact_digest",
        "provider_release_digest",
        "logical_call_id",
        "proposal_generation_session_id",
        "diagnostic_slot_id",
    }
    if {
        key: value
        for key, value in exact9.items()
        if key not in allowed_exact_delta
    } != {
        key: value
        for key, value in exact8.items()
        if key not in allowed_exact_delta
    }:
        raise Wave2IntegrationError("V9 changed the V8 Provider request contract")
    release8 = deepcopy(_read_json(repo_root / V8_RELEASE_REL))
    release9 = deepcopy(release)
    for value in (release8, release9):
        value.pop("schema", None)
        value.pop("release_digest", None)
        value.pop("availability_predecessor", None)
    if release9 != release8:
        raise Wave2IntegrationError("V9 changed the V8 scientific release")
    policy8 = deepcopy(_read_json(repo_root / V8_POLICY_REL))
    policy9 = deepcopy(policy)
    for value in (policy8, policy9):
        value.pop("schema", None)
        value.pop("inherited_v7_policy_ref", None)
        value.pop("inherited_v7_policy_digest", None)
        value.pop("inherited_v8_policy_ref", None)
        value.pop("inherited_v8_policy_digest", None)
        value["diagnostic_slot"].pop("slot_id", None)
    if policy9 != policy8:
        raise Wave2IntegrationError("V9 changed bounded retry semantics")
    if (
        manifest["future_r1_scientific_contract"]
        != v8["future_r1_scientific_contract"]
        or manifest["response_contract_equivalence"]
        != v8["response_contract_equivalence"]
        or manifest["preserved_scientific_identity"]
        != v8["preserved_scientific_identity"]
        or manifest["provider_returned_model_evidence"]
        != v8["provider_returned_model_evidence"]
        or exact9["request_payload_digest"]
        != exact_v8_probe_request_payload_digest(repo_root)
    ):
        raise Wave2IntegrationError("V9 changed the V8 R1 scientific contract")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V9 is not pre-outcome")
    return manifest


def provider_free_v9_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v9(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v9-dry-run-receipt.v1",
        "status": "PASS_PROVIDER_FREE_V9_AVAILABILITY_RECHECK",
        "attempt_id": V9_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_through_v8_seals_verified": True,
        "v8_scientific_contract_exactly_preserved": True,
        "v8_request_payload_digest_exactly_preserved": True,
        "fresh_attempt_call_session_db_identities_verified": True,
        "bounded_retry_semantics_exactly_preserved": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def prefreeze_v9_runtime_spec() -> dict[str, Any]:
    spec = prefreeze_v8_runtime_spec()
    spec.update(
        {
            "label": "V9",
            "attempt_id": V9_ATTEMPT_ID,
            "attempt_rel": V9_ATTEMPT_RECEIPT_REL,
            "attempt_schema": (
                "recclaw.research-line.prefreeze-v9-provider-attempt-receipt.v1"
            ),
            "auth_rel": V9_AUTH_REL,
            "blocked_rel": V9_BLOCKED_REL,
            "blocked_schema": (
                "recclaw.research-line.prefreeze-v9-blocked-receipt.v1"
            ),
            "dry_run_rel": V9_DRY_RUN_REL,
            "manifest_rel": V9_MANIFEST_REL,
            "policy_rel": V9_POLICY_REL,
            "private_root": V9_PRIVATE_ROOT,
            "ready_rel": V9_READY_REL,
            "ready_schema": (
                "recclaw.research-line.r1-prefreeze-ready-receipt.v9"
            ),
            "release_rel": V9_RELEASE_REL,
            "verification_rel": V9_VERIFICATION_REL,
            "verification_schema": (
                "recclaw.research-line.prefreeze-v9-verification-receipt.v1"
            ),
            "validate": validate_prefreeze_v9,
            "dry_run": provider_free_v9_dry_run,
            "pass_status": "PASS_PROVIDER_FREE_V9_AVAILABILITY_RECHECK",
            "verify_predecessor": verify_v8_seal,
            "expected_policy": expected_v9_retry_policy,
            "expected_manifest": expected_prefreeze_v9_manifest,
            "expected_authorization": expected_v9_authorization,
            "expected_release": expected_v9_provider_release,
            "logical_call_id": V9_LOGICAL_CALL_ID,
            "session_id": V9_SESSION_ID,
            "slot_id": "PREFREEZE_V9_AVAILABILITY_RECHECK",
            "physical_root": v9_physical_root,
            "hard_blocked_on_transient_exhaustion": True,
        }
    )
    return spec


def v10_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V10 physical attempt ordinal must be 1..3")
    return V10_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v9_seal(repo_root: Path) -> dict[str, str]:
    verified = verify_v8_seal(repo_root)
    for relative, digest in V9_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V9 bytes changed: {relative.as_posix()}"
            )
        verified[relative.as_posix()] = digest
    return verified


def exact_v10_probe_request_payload(repo_root: Path) -> dict[str, Any]:
    payload = exact_v8_probe_request_payload(repo_root)
    payload["model"] = V10_REQUEST_MODEL_ALIAS
    return payload


def exact_v10_probe_request_payload_digest(repo_root: Path) -> str:
    return sha256_digest(exact_v10_probe_request_payload(repo_root))


def validate_v10_route_snapshot_pair(
    *, requested_model: str, returned_model: str
) -> None:
    if requested_model != V10_REQUEST_MODEL_ALIAS:
        raise Wave2IntegrationError("V10 request routing alias mismatch")
    if returned_model != V8_MODEL_SNAPSHOT:
        raise Wave2IntegrationError("V10 returned snapshot mismatch")


def expected_v10_broker_release(repo_root: Path) -> dict[str, Any]:
    release = expected_v8_broker_release(repo_root)
    release.pop("release_digest")
    release["model"] = V10_REQUEST_MODEL_ALIAS
    return {**release, "release_digest": sha256_digest(release)}


def expected_v10_provider_release(repo_root: Path) -> dict[str, Any]:
    verify_v9_seal(repo_root)
    release = deepcopy(_read_json(repo_root / V9_RELEASE_REL))
    release.pop("release_digest")
    release.pop("availability_predecessor", None)
    broker_release = expected_v10_broker_release(repo_root)
    release.update(
        {
            "schema": "recclaw.research-line.provider-release-contract.v10",
            "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
            "required_returned_snapshot": V8_MODEL_SNAPSHOT,
            "request_and_return_matching": (
                "EXACT_GATEWAY_ROUTE_ALIAS_TO_EXACT_RETURNED_SNAPSHOT"
            ),
            "alias_request_or_fallback": (
                "EXACT_ROUTING_ALIAS_ONLY_REQUEST_MODEL_FALLBACK_FORBIDDEN"
            ),
            "transport_release_ref": "INLINE_LAB_API_BROKER_RELEASE_V1_V10",
            "transport_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(broker_release)
            ),
            "transport_release_contract_digest": broker_release[
                "release_digest"
            ],
            "routing_correction_predecessor": {
                "v9_release_digest": V9_SEALED_DIGESTS[V9_RELEASE_REL],
                "v9_attempt_digest": V9_SEALED_DIGESTS[
                    V9_ATTEMPT_RECEIPT_REL
                ],
                "v9_blocked_digest": V9_SEALED_DIGESTS[V9_BLOCKED_REL],
                "observed_supported_alias_source": "SEALED_V7_HTTP200",
                "snapshot_literal_route_failures": 6,
            },
        }
    )
    release["model_identity_contract_digest"] = (
        _model_identity_contract_digest(release)
    )
    return {**release, "release_digest": sha256_digest(release)}


def expected_v10_retry_policy(repo_root: Path) -> dict[str, Any]:
    verify_v9_seal(repo_root)
    policy = deepcopy(_read_json(repo_root / V9_POLICY_REL))
    policy["schema"] = "recclaw.research-line.r1-provider-retry-policy.v10"
    policy.pop("inherited_v8_policy_ref", None)
    policy.pop("inherited_v8_policy_digest", None)
    policy["inherited_v9_policy_ref"] = _repo_ref(V9_POLICY_REL)
    policy["inherited_v9_policy_digest"] = V9_SEALED_DIGESTS[V9_POLICY_REL]
    policy["diagnostic_slot"]["slot_id"] = (
        "PREFREEZE_V10_MODEL_ROUTING_CORRECTION"
    )
    return policy


def expected_prefreeze_v10_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v9_seal(repo_root)
    v9 = _read_json(repo_root / V9_MANIFEST_REL)
    release = expected_v10_provider_release(repo_root)
    policy = expected_v10_retry_policy(repo_root)
    manifest = deepcopy(v9)
    manifest["schema"] = "recclaw.research-line.r1-r2-prefreeze-attempt.v10"
    manifest["attempt_identity"] = {
        "attempt_id": V10_ATTEMPT_ID,
        "base_commit": V9_HEAD,
        "base_parent": V9_PARENT,
        "base_tree": V9_TREE,
        "pre_outcome": True,
        "distinct_from_v1_v2_v3_v4_v5_v6_v7_v8_v9_attempts": True,
        "old_attempt_call_session_db_identity_reuse": False,
    }
    manifest["sealed_predecessor_evidence"] = {
        "v9_manifest_digest": V9_SEALED_DIGESTS[V9_MANIFEST_REL],
        "v9_attempt_digest": V9_SEALED_DIGESTS[V9_ATTEMPT_RECEIPT_REL],
        "v9_blocked_digest": V9_SEALED_DIGESTS[V9_BLOCKED_REL],
        "v1_through_v9_preservation": (
            "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
        ),
    }
    manifest["authorized_v10_model_routing_correction"] = {
        "request_model_literal": V10_REQUEST_MODEL_ALIAS,
        "request_literal_role": "GATEWAY_ROUTING_INPUT_ONLY",
        "required_returned_snapshot": V8_MODEL_SNAPSHOT,
        "actual_provider_envelope_model_required": True,
        "request_model_fallback_prefix_regex_startswith_allowlist": (
            "FORBIDDEN"
        ),
        "only_scientific_delta": "GATEWAY_ROUTE_LITERAL",
        "accepted_snapshot_or_comparison_change": False,
        "v11_or_unbounded_retry_authorized": False,
    }
    manifest["provider_release_contract"] = release
    exact = manifest["exact_provider_contract"]
    exact.update(
        {
            "model": V10_REQUEST_MODEL_ALIAS,
            "model_digest": sha256_digest(
                {"model": V10_REQUEST_MODEL_ALIAS}
            ),
            "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
            "required_returned_snapshot": V8_MODEL_SNAPSHOT,
            "model_identity_pair_digest": sha256_digest(
                {
                    "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
                    "required_returned_snapshot": V8_MODEL_SNAPSHOT,
                }
            ),
            "transport_provider_release_digest": release[
                "transport_release_contract_digest"
            ],
            "provider_release_ref": _repo_ref(V10_RELEASE_REL),
            "provider_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(release)
            ),
            "provider_release_digest": release["release_digest"],
            "request_payload_digest": exact_v10_probe_request_payload_digest(
                repo_root
            ),
            "logical_call_id": V10_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V10_SESSION_ID,
            "diagnostic_slot_id": (
                "PREFREEZE_V10_MODEL_ROUTING_CORRECTION"
            ),
        }
    )
    identities = []
    for ordinal in (1, 2, 3):
        private_digest = sha256_digest(
            {"path": v10_physical_root(ordinal).as_posix()}
        )
        identities.append(
            {
                "ordinal": ordinal,
                "physical_attempt_identity_digest": sha256_digest(
                    {
                        "attempt_id": V10_ATTEMPT_ID,
                        "diagnostic_slot": (
                            "PREFREEZE_V10_MODEL_ROUTING_CORRECTION"
                        ),
                        "ordinal": ordinal,
                        "private_root_digest": private_digest,
                    }
                ),
                "private_root_digest": private_digest,
            }
        )
    manifest["bounded_retry"].update(
        {
            "policy_ref": _repo_ref(V10_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "physical_attempt_identities": identities,
        }
    )
    future = manifest["future_r1_scientific_contract"]
    shared = sha256_digest(
        {
            "inherited_v9_shared_call_contract_digest": future[
                "shared_call_contract_digest"
            ],
            "v10_provider_release_contract_digest": release[
                "release_digest"
            ],
        }
    )
    future.update(
        {
            "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
            "required_returned_snapshot": V8_MODEL_SNAPSHOT,
            "shared_call_contract_digest": shared,
            "side_a_call_contract_digest": shared,
            "side_b_call_contract_digest": shared,
        }
    )
    manifest["inherited_v9_scientific_contract_digest"] = sha256_digest(
        v9["future_r1_scientific_contract"]
    )
    manifest["v10_scientific_contract_digest"] = sha256_digest(future)
    manifest["r1_worker_launch_authorized"] = False
    return manifest


def expected_v10_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v10_manifest(repo_root)
    policy = expected_v10_retry_policy(repo_root)
    release = expected_v10_provider_release(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v10-authorization.v1",
        "status": "AUTHORIZED_ONE_V10_MODEL_ROUTING_DIAGNOSTIC_SLOT",
        "attempt_id": V10_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V10_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V10_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "provider_release_ref": _repo_ref(V10_RELEASE_REL),
        "provider_release_artifact_digest": bytes_sha256(
            canonical_json_bytes(release)
        ),
        "provider_release_contract_digest": release["release_digest"],
        "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
        "required_returned_snapshot": V8_MODEL_SNAPSHOT,
        "diagnostic_token_ceiling": V8_DIAGNOSTIC_TOKEN_CEILING,
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v10(repo_root: Path) -> dict[str, Any]:
    verify_v9_seal(repo_root)
    release = _load_exact(
        repo_root / V10_RELEASE_REL,
        expected_v10_provider_release(repo_root),
    )
    policy = _load_exact(
        repo_root / V10_POLICY_REL, expected_v10_retry_policy(repo_root)
    )
    manifest = _load_exact(
        repo_root / V10_MANIFEST_REL,
        expected_prefreeze_v10_manifest(repo_root),
    )
    _load_exact(
        repo_root / V10_AUTH_REL, expected_v10_authorization(repo_root)
    )
    v9 = _read_json(repo_root / V9_MANIFEST_REL)
    exact9 = v9["exact_provider_contract"]
    exact10 = manifest["exact_provider_contract"]
    validate_v10_route_snapshot_pair(
        requested_model=exact10["requested_model_literal"],
        returned_model=exact10["required_returned_snapshot"],
    )
    allowed_exact_delta = {
        "model",
        "model_digest",
        "requested_model_literal",
        "model_identity_pair_digest",
        "transport_provider_release_digest",
        "provider_release_ref",
        "provider_release_artifact_digest",
        "provider_release_digest",
        "request_payload_digest",
        "logical_call_id",
        "proposal_generation_session_id",
        "diagnostic_slot_id",
    }
    if {
        key: value
        for key, value in exact10.items()
        if key not in allowed_exact_delta
    } != {
        key: value
        for key, value in exact9.items()
        if key not in allowed_exact_delta
    }:
        raise Wave2IntegrationError("V10 changed a non-routing Provider field")
    future9 = deepcopy(v9["future_r1_scientific_contract"])
    future10 = deepcopy(manifest["future_r1_scientific_contract"])
    for field in (
        "requested_model_literal",
        "shared_call_contract_digest",
        "side_a_call_contract_digest",
        "side_b_call_contract_digest",
    ):
        future9.pop(field, None)
        future10.pop(field, None)
    if (
        future10 != future9
        or manifest["response_contract_equivalence"]
        != v9["response_contract_equivalence"]
        or manifest["preserved_scientific_identity"]
        != v9["preserved_scientific_identity"]
        or manifest["provider_returned_model_evidence"]
        != v9["provider_returned_model_evidence"]
        or exact10["request_payload_digest"]
        != exact_v10_probe_request_payload_digest(repo_root)
    ):
        raise Wave2IntegrationError("V10 changed the preserved R1 contract")
    policy9 = deepcopy(_read_json(repo_root / V9_POLICY_REL))
    policy10 = deepcopy(policy)
    for value in (policy9, policy10):
        value.pop("schema", None)
        value.pop("inherited_v8_policy_ref", None)
        value.pop("inherited_v8_policy_digest", None)
        value.pop("inherited_v9_policy_ref", None)
        value.pop("inherited_v9_policy_digest", None)
        value["diagnostic_slot"].pop("slot_id", None)
    if policy10 != policy9:
        raise Wave2IntegrationError("V10 changed retry semantics")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V10 is not pre-outcome")
    return manifest


def provider_free_v10_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v10(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v10-dry-run-receipt.v1",
        "status": "PASS_PROVIDER_FREE_V10_MODEL_ROUTING_CORRECTION",
        "attempt_id": V10_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_through_v9_seals_verified": True,
        "exact_gateway_alias_request_verified": True,
        "exact_actual_envelope_snapshot_required": True,
        "request_model_fallback_forbidden": True,
        "non_routing_scientific_contract_exactly_preserved": True,
        "bounded_retry_semantics_exactly_preserved": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def prefreeze_v10_runtime_spec() -> dict[str, Any]:
    spec = prefreeze_v9_runtime_spec()
    spec.update(
        {
            "label": "V10",
            "attempt_id": V10_ATTEMPT_ID,
            "attempt_rel": V10_ATTEMPT_RECEIPT_REL,
            "attempt_schema": (
                "recclaw.research-line.prefreeze-v10-provider-attempt-receipt.v1"
            ),
            "auth_rel": V10_AUTH_REL,
            "blocked_rel": V10_BLOCKED_REL,
            "blocked_schema": (
                "recclaw.research-line.prefreeze-v10-blocked-receipt.v1"
            ),
            "dry_run_rel": V10_DRY_RUN_REL,
            "manifest_rel": V10_MANIFEST_REL,
            "policy_rel": V10_POLICY_REL,
            "private_root": V10_PRIVATE_ROOT,
            "ready_rel": V10_READY_REL,
            "ready_schema": (
                "recclaw.research-line.r1-prefreeze-ready-receipt.v10"
            ),
            "release_rel": V10_RELEASE_REL,
            "verification_rel": V10_VERIFICATION_REL,
            "verification_schema": (
                "recclaw.research-line.prefreeze-v10-verification-receipt.v1"
            ),
            "validate": validate_prefreeze_v10,
            "dry_run": provider_free_v10_dry_run,
            "requested_model": V10_REQUEST_MODEL_ALIAS,
            "required_returned_model": V8_MODEL_SNAPSHOT,
            "validate_model_pair": validate_v10_route_snapshot_pair,
            "identity_receipt_fields": {
                "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
                "required_returned_snapshot": V8_MODEL_SNAPSHOT,
            },
            "blocked_identity_field": (
                "exact_gateway_route_alias_and_returned_snapshot"
            ),
            "pass_classification": (
                "PASS_EXACT_ALIAS_ROUTE_SNAPSHOT_AUTH_PROVIDER_LOCAL_SCHEMA"
            ),
            "pass_status": (
                "PASS_PROVIDER_FREE_V10_MODEL_ROUTING_CORRECTION"
            ),
            "verify_predecessor": verify_v9_seal,
            "expected_policy": expected_v10_retry_policy,
            "expected_manifest": expected_prefreeze_v10_manifest,
            "expected_authorization": expected_v10_authorization,
            "expected_release": expected_v10_provider_release,
            "logical_call_id": V10_LOGICAL_CALL_ID,
            "session_id": V10_SESSION_ID,
            "slot_id": "PREFREEZE_V10_MODEL_ROUTING_CORRECTION",
            "physical_root": v10_physical_root,
            "hard_blocked_on_transient_exhaustion": False,
        }
    )
    return spec


def v11_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V11 physical attempt ordinal must be 1..3")
    return V11_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v10_seal(repo_root: Path) -> dict[str, str]:
    verified = verify_v9_seal(repo_root)
    for relative, digest in V10_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V10 bytes changed: {relative.as_posix()}"
            )
        verified[relative.as_posix()] = digest
    return verified


def expected_v11_provider_release(repo_root: Path) -> dict[str, Any]:
    """Bind the supported gpt-5.4 route; returned model is observation only."""

    verify_v10_seal(repo_root)
    release = deepcopy(_read_json(repo_root / V10_RELEASE_REL))
    release.pop("release_digest")
    release.pop("model_identity_contract_digest")
    release.pop("routing_correction_predecessor", None)
    release.pop("required_returned_snapshot", None)
    release.update(
        {
            "schema": "recclaw.research-line.provider-release-contract.v11",
            "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
            "request_and_return_matching": "EXACT_REQUEST_ROUTE_ONLY",
            "returned_model_evidence_policy": "OBSERVED_METADATA_NON_BLOCKING",
            "alias_request_or_fallback": "EXACT_GPT_5_4_REQUEST_NO_FALLBACK",
            "engineering_predecessor": {
                "v10_release_digest": V10_SEALED_DIGESTS[V10_RELEASE_REL],
                "v10_attempt_digest": V10_SEALED_DIGESTS[
                    V10_ATTEMPT_RECEIPT_REL
                ],
                "v10_blocked_digest": V10_SEALED_DIGESTS[V10_BLOCKED_REL],
                "v10_functional_http_status": 200,
            },
        }
    )
    identity_fields = {
        key: release[key]
        for key in _MODEL_IDENTITY_BINDING_FIELDS
        if key != "required_returned_snapshot"
    }
    identity_fields["returned_model_evidence_policy"] = release[
        "returned_model_evidence_policy"
    ]
    release["model_identity_contract_digest"] = sha256_digest(identity_fields)
    return {**release, "release_digest": sha256_digest(release)}


def expected_v11_retry_policy(repo_root: Path) -> dict[str, Any]:
    verify_v10_seal(repo_root)
    policy = deepcopy(_read_json(repo_root / V10_POLICY_REL))
    policy["schema"] = "recclaw.research-line.r1-provider-retry-policy.v11"
    policy.pop("inherited_v9_policy_ref", None)
    policy.pop("inherited_v9_policy_digest", None)
    policy["inherited_v10_policy_ref"] = _repo_ref(V10_POLICY_REL)
    policy["inherited_v10_policy_digest"] = V10_SEALED_DIGESTS[V10_POLICY_REL]
    policy["diagnostic_slot"]["slot_id"] = (
        "PREFREEZE_V11_ENGINEERING_VALIDATION"
    )
    return policy


def expected_prefreeze_v11_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v10_seal(repo_root)
    v10 = _read_json(repo_root / V10_MANIFEST_REL)
    release = expected_v11_provider_release(repo_root)
    policy = expected_v11_retry_policy(repo_root)
    manifest = deepcopy(v10)
    manifest["schema"] = "recclaw.research-line.r1-r2-prefreeze-attempt.v11"
    manifest["attempt_identity"] = {
        "attempt_id": V11_ATTEMPT_ID,
        "base_commit": V10_HEAD,
        "base_parent": V10_PARENT,
        "base_tree": V10_TREE,
        "pre_outcome": True,
        "distinct_from_v1_through_v10_attempts": True,
        "old_attempt_call_session_db_identity_reuse": False,
    }
    manifest["sealed_predecessor_evidence"] = {
        "v10_manifest_digest": V10_SEALED_DIGESTS[V10_MANIFEST_REL],
        "v10_attempt_digest": V10_SEALED_DIGESTS[V10_ATTEMPT_RECEIPT_REL],
        "v10_blocked_digest": V10_SEALED_DIGESTS[V10_BLOCKED_REL],
        "v1_through_v10_preservation": (
            "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
        ),
    }
    manifest.pop("authorized_v10_model_routing_correction", None)
    manifest["authorized_v11_engineering_validation"] = {
        "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
        "returned_model_evidence_policy": "OBSERVED_METADATA_NON_BLOCKING",
        "functional_gates": [
            "REAL_PROVIDER_ROUTE_EXECUTES",
            "STRICT_SCHEMA_AND_LOCAL_SEMANTICS_PASS",
            "OPEN_RESEARCH_TARGET_PRESERVED",
            "NO_FIXED_66_TUNING_STATIC_OR_FALLBACK_DEGRADATION",
        ],
        "candidate_training_outcome_held_out": "FORBIDDEN_IN_PREFREEZE",
    }
    manifest["downstream_task_opening_acceptance_criteria"] = [
        "FUNCTION_IS_REAL_AND_RUNNABLE",
        "END_TO_END_RESULT_CHAIN_IS_REAL_AND_VALID",
        "IMPLEMENTATION_PRECISELY_SERVES_RECCLAW_RESEARCH_TARGET_AND_REQUIRED_EFFECT",
        "NO_FIXED_66_CONFIG_TUNING_STATIC_CANDIDATE_LOW_CHANGE_WRAPPER_FALLBACK_OR_MOCK_SMOKE_SUBSTITUTION",
    ]
    manifest["provider_release_contract"] = release
    exact = manifest["exact_provider_contract"]
    exact.pop("required_returned_snapshot", None)
    exact.pop("model_identity_pair_digest", None)
    exact.update(
        {
            "model": V10_REQUEST_MODEL_ALIAS,
            "model_digest": sha256_digest({"model": V10_REQUEST_MODEL_ALIAS}),
            "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
            "returned_model_evidence_policy": "OBSERVED_METADATA_NON_BLOCKING",
            "provider_release_ref": _repo_ref(V11_RELEASE_REL),
            "provider_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(release)
            ),
            "provider_release_digest": release["release_digest"],
            "request_payload_digest": exact_v10_probe_request_payload_digest(
                repo_root
            ),
            "logical_call_id": V11_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V11_SESSION_ID,
            "diagnostic_slot_id": "PREFREEZE_V11_ENGINEERING_VALIDATION",
        }
    )
    identities = []
    for ordinal in (1, 2, 3):
        private_digest = sha256_digest(
            {"path": v11_physical_root(ordinal).as_posix()}
        )
        identities.append(
            {
                "ordinal": ordinal,
                "physical_attempt_identity_digest": sha256_digest(
                    {
                        "attempt_id": V11_ATTEMPT_ID,
                        "diagnostic_slot": "PREFREEZE_V11_ENGINEERING_VALIDATION",
                        "ordinal": ordinal,
                        "private_root_digest": private_digest,
                    }
                ),
                "private_root_digest": private_digest,
            }
        )
    manifest["bounded_retry"].update(
        {
            "policy_ref": _repo_ref(V11_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "physical_attempt_identities": identities,
        }
    )
    future = manifest["future_r1_scientific_contract"]
    future.pop("required_returned_snapshot", None)
    future["returned_model_evidence_policy"] = "OBSERVED_METADATA_NON_BLOCKING"
    shared = sha256_digest(
        {
            "inherited_v10_shared_call_contract_digest": future[
                "shared_call_contract_digest"
            ],
            "v11_provider_release_contract_digest": release["release_digest"],
        }
    )
    future["shared_call_contract_digest"] = shared
    future["side_a_call_contract_digest"] = shared
    future["side_b_call_contract_digest"] = shared
    manifest["inherited_v10_scientific_contract_digest"] = sha256_digest(
        v10["future_r1_scientific_contract"]
    )
    manifest["v11_scientific_contract_digest"] = sha256_digest(future)
    manifest["r1_worker_launch_authorized"] = False
    return manifest


def expected_v11_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v11_manifest(repo_root)
    policy = expected_v11_retry_policy(repo_root)
    release = expected_v11_provider_release(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v11-authorization.v1",
        "status": "AUTHORIZED_ONE_V11_ENGINEERING_VALIDATION_SLOT",
        "attempt_id": V11_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V11_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V11_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "provider_release_ref": _repo_ref(V11_RELEASE_REL),
        "provider_release_artifact_digest": bytes_sha256(
            canonical_json_bytes(release)
        ),
        "provider_release_contract_digest": release["release_digest"],
        "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
        "returned_model_evidence_policy": "OBSERVED_METADATA_NON_BLOCKING",
        "diagnostic_token_ceiling": V8_DIAGNOSTIC_TOKEN_CEILING,
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v11(repo_root: Path) -> dict[str, Any]:
    verify_v10_seal(repo_root)
    _load_exact(
        repo_root / V11_RELEASE_REL, expected_v11_provider_release(repo_root)
    )
    policy = _load_exact(
        repo_root / V11_POLICY_REL, expected_v11_retry_policy(repo_root)
    )
    manifest = _load_exact(
        repo_root / V11_MANIFEST_REL, expected_prefreeze_v11_manifest(repo_root)
    )
    _load_exact(
        repo_root / V11_AUTH_REL, expected_v11_authorization(repo_root)
    )
    v10 = _read_json(repo_root / V10_MANIFEST_REL)
    if (
        manifest["preserved_scientific_identity"]
        != v10["preserved_scientific_identity"]
        or manifest["response_contract_equivalence"]
        != v10["response_contract_equivalence"]
    ):
        raise Wave2IntegrationError("V11 changed the research implementation target")
    future10 = deepcopy(v10["future_r1_scientific_contract"])
    future11 = deepcopy(manifest["future_r1_scientific_contract"])
    for value in (future10, future11):
        value.pop("required_returned_snapshot", None)
        value.pop("returned_model_evidence_policy", None)
        value.pop("shared_call_contract_digest", None)
        value.pop("side_a_call_contract_digest", None)
        value.pop("side_b_call_contract_digest", None)
    if future10 != future11:
        raise Wave2IntegrationError("V11 changed the functional R1 comparison")
    if (
        manifest["future_r1_scientific_contract"][
            "fixed_66_or_static_candidate_fallback"
        ]
        != "FORBIDDEN"
        or any(manifest["pre_outcome_counters"].values())
    ):
        raise Wave2IntegrationError("V11 degraded or crossed the pre-outcome gate")
    policy10 = deepcopy(_read_json(repo_root / V10_POLICY_REL))
    policy11 = deepcopy(policy)
    for value in (policy10, policy11):
        value.pop("schema", None)
        value.pop("inherited_v9_policy_ref", None)
        value.pop("inherited_v9_policy_digest", None)
        value.pop("inherited_v10_policy_ref", None)
        value.pop("inherited_v10_policy_digest", None)
        value["diagnostic_slot"].pop("slot_id", None)
    if policy10 != policy11:
        raise Wave2IntegrationError("V11 changed retry semantics")
    return manifest


def provider_free_v11_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v11(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v11-dry-run-receipt.v1",
        "status": "PASS_PROVIDER_FREE_V11_ENGINEERING_VALIDATION",
        "attempt_id": V11_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_through_v10_seals_verified": True,
        "gpt_5_4_route_fixed": True,
        "returned_model_metadata_non_blocking": True,
        "strict_schema_and_local_uniqueness_fail_closed": True,
        "open_research_target_preserved": True,
        "no_fixed_66_tuning_static_or_fallback_degradation": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def prefreeze_v11_runtime_spec() -> dict[str, Any]:
    spec = prefreeze_v10_runtime_spec()
    spec.update(
        {
            "label": "V11",
            "attempt_id": V11_ATTEMPT_ID,
            "attempt_rel": V11_ATTEMPT_RECEIPT_REL,
            "attempt_schema": (
                "recclaw.research-line.prefreeze-v11-provider-attempt-receipt.v1"
            ),
            "auth_rel": V11_AUTH_REL,
            "blocked_rel": V11_BLOCKED_REL,
            "blocked_schema": (
                "recclaw.research-line.prefreeze-v11-blocked-receipt.v1"
            ),
            "dry_run_rel": V11_DRY_RUN_REL,
            "manifest_rel": V11_MANIFEST_REL,
            "policy_rel": V11_POLICY_REL,
            "private_root": V11_PRIVATE_ROOT,
            "ready_rel": V11_READY_REL,
            "ready_schema": (
                "recclaw.research-line.r1-prefreeze-ready-receipt.v11"
            ),
            "release_rel": V11_RELEASE_REL,
            "verification_rel": V11_VERIFICATION_REL,
            "verification_schema": (
                "recclaw.research-line.prefreeze-v11-verification-receipt.v1"
            ),
            "validate": validate_prefreeze_v11,
            "dry_run": provider_free_v11_dry_run,
            "requested_model": V10_REQUEST_MODEL_ALIAS,
            "required_returned_model": None,
            "enforce_returned_model": False,
            "identity_receipt_fields": {
                "requested_model_literal": V10_REQUEST_MODEL_ALIAS,
                "returned_model_evidence_policy": "OBSERVED_METADATA_NON_BLOCKING",
            },
            "blocked_identity_field": "gpt_5_4_route_and_functional_contract",
            "pass_classification": (
                "PASS_GPT_5_4_FUNCTIONAL_SCHEMA_AND_LOCAL_CONTRACT"
            ),
            "pass_status": "PASS_PROVIDER_FREE_V11_ENGINEERING_VALIDATION",
            "verify_predecessor": verify_v10_seal,
            "expected_policy": expected_v11_retry_policy,
            "expected_manifest": expected_prefreeze_v11_manifest,
            "expected_authorization": expected_v11_authorization,
            "expected_release": expected_v11_provider_release,
            "logical_call_id": V11_LOGICAL_CALL_ID,
            "session_id": V11_SESSION_ID,
            "slot_id": "PREFREEZE_V11_ENGINEERING_VALIDATION",
            "physical_root": v11_physical_root,
            "hard_blocked_on_transient_exhaustion": False,
        }
    )
    spec.pop("validate_model_pair", None)
    return spec


__all__ = [
    "BROKER_SOURCE_REL",
    "V5_ATTEMPT_ID",
    "V5_ATTEMPT_RECEIPT_REL",
    "V5_ATTEMPT_RECEIPT_SCHEMA",
    "V5_AUTH_REL",
    "V5_BLOCKED_REL",
    "V5_BLOCKED_SCHEMA",
    "V5_DRY_RUN_REL",
    "V5_LOGICAL_CALL_ID",
    "V5_MANIFEST_REL",
    "V5_POLICY_REL",
    "V5_PRIVATE_ROOT",
    "V5_READY_REL",
    "V5_READY_SCHEMA",
    "V5_SESSION_ID",
    "V5_VERIFICATION_REL",
    "V5_VERIFICATION_SCHEMA",
    "diagnostic_reason_vocabulary",
    "expected_prefreeze_v5_manifest",
    "expected_v5_authorization",
    "expected_v5_retry_policy",
    "provider_free_v5_dry_run",
    "v5_physical_root",
    "validate_prefreeze_v5",
    "verify_v4_seal",
    "V5_SEALED_DIGESTS",
    "V6_ATTEMPT_ID",
    "V6_ATTEMPT_RECEIPT_REL",
    "V6_ATTEMPT_RECEIPT_SCHEMA",
    "V6_AUTH_REL",
    "V6_BLOCKED_REL",
    "V6_BLOCKED_SCHEMA",
    "V6_DRY_RUN_REL",
    "V6_LOGICAL_CALL_ID",
    "V6_MANIFEST_REL",
    "V6_POLICY_REL",
    "V6_PRIVATE_ROOT",
    "V6_READY_REL",
    "V6_READY_SCHEMA",
    "V6_RELEASE_REL",
    "V6_REQUESTED_MODEL_ALIAS",
    "V6_REQUIRED_RETURNED_SNAPSHOT",
    "V6_SESSION_ID",
    "V6_VERIFICATION_REL",
    "V6_VERIFICATION_SCHEMA",
    "expected_prefreeze_v6_manifest",
    "expected_v6_authorization",
    "expected_v6_provider_release",
    "expected_v6_retry_policy",
    "provider_free_v6_dry_run",
    "v6_physical_root",
    "validate_prefreeze_v6",
    "validate_v6_exact_model_pair",
    "verify_v5_seal",
    "V6_SEALED_DIGESTS",
    "V7_ATTEMPT_ID",
    "V7_ATTEMPT_RECEIPT_REL",
    "V7_ATTEMPT_RECEIPT_SCHEMA",
    "V7_AUTH_REL",
    "V7_BLOCKED_REL",
    "V7_BLOCKED_SCHEMA",
    "V7_DIAGNOSTIC_TOKEN_CEILING",
    "V7_DRY_RUN_REL",
    "V7_LOGICAL_CALL_ID",
    "V7_MANIFEST_REL",
    "V7_POLICY_REL",
    "V7_PRIVATE_ROOT",
    "V7_READY_REL",
    "V7_READY_SCHEMA",
    "V7_RELEASE_REL",
    "V7_SESSION_ID",
    "V7_VERIFICATION_REL",
    "V7_VERIFICATION_SCHEMA",
    "exact_v7_probe_request_payload_digest",
    "expected_prefreeze_v7_manifest",
    "expected_v7_authorization",
    "expected_v7_provider_release",
    "expected_v7_retry_policy",
    "provider_free_v7_dry_run",
    "v7_physical_root",
    "validate_prefreeze_v7",
    "verify_v6_seal",
    "V7_SEALED_DIGESTS",
    "V8_ATTEMPT_ID",
    "V8_ATTEMPT_RECEIPT_REL",
    "V8_ATTEMPT_RECEIPT_SCHEMA",
    "V8_AUTH_REL",
    "V8_BLOCKED_REL",
    "V8_BLOCKED_SCHEMA",
    "V8_DIAGNOSTIC_TOKEN_CEILING",
    "V8_DRY_RUN_REL",
    "V8_LOGICAL_CALL_ID",
    "V8_MANIFEST_REL",
    "V8_MODEL_SNAPSHOT",
    "V8_POLICY_REL",
    "V8_PRIVATE_ROOT",
    "V8_READY_REL",
    "V8_READY_SCHEMA",
    "V8_RELEASE_REL",
    "V8_SESSION_ID",
    "V8_VERIFICATION_REL",
    "V8_VERIFICATION_SCHEMA",
    "exact_v8_probe_request_payload",
    "exact_v8_probe_request_payload_digest",
    "expected_prefreeze_v8_manifest",
    "expected_v8_authorization",
    "expected_v8_broker_release",
    "expected_v8_provider_release",
    "expected_v8_retry_policy",
    "prefreeze_v8_runtime_spec",
    "provider_free_v8_dry_run",
    "v8_physical_root",
    "validate_prefreeze_v8",
    "validate_v8_exact_snapshot_pair",
    "verify_v7_seal",
    "V8_SEALED_DIGESTS",
    "V9_ATTEMPT_ID",
    "V9_ATTEMPT_RECEIPT_REL",
    "V9_AUTH_REL",
    "V9_BLOCKED_REL",
    "V9_DRY_RUN_REL",
    "V9_LOGICAL_CALL_ID",
    "V9_MANIFEST_REL",
    "V9_POLICY_REL",
    "V9_PRIVATE_ROOT",
    "V9_READY_REL",
    "V9_RELEASE_REL",
    "V9_SESSION_ID",
    "V9_VERIFICATION_REL",
    "expected_prefreeze_v9_manifest",
    "expected_v9_authorization",
    "expected_v9_provider_release",
    "expected_v9_retry_policy",
    "prefreeze_v9_runtime_spec",
    "provider_free_v9_dry_run",
    "v9_physical_root",
    "validate_prefreeze_v9",
    "verify_v8_seal",
    "V9_SEALED_DIGESTS",
    "V10_ATTEMPT_ID",
    "V10_ATTEMPT_RECEIPT_REL",
    "V10_AUTH_REL",
    "V10_BLOCKED_REL",
    "V10_DRY_RUN_REL",
    "V10_LOGICAL_CALL_ID",
    "V10_MANIFEST_REL",
    "V10_POLICY_REL",
    "V10_PRIVATE_ROOT",
    "V10_READY_REL",
    "V10_RELEASE_REL",
    "V10_REQUEST_MODEL_ALIAS",
    "V10_SESSION_ID",
    "V10_VERIFICATION_REL",
    "exact_v10_probe_request_payload",
    "exact_v10_probe_request_payload_digest",
    "expected_prefreeze_v10_manifest",
    "expected_v10_authorization",
    "expected_v10_broker_release",
    "expected_v10_provider_release",
    "expected_v10_retry_policy",
    "prefreeze_v10_runtime_spec",
    "provider_free_v10_dry_run",
    "v10_physical_root",
    "validate_prefreeze_v10",
    "validate_v10_route_snapshot_pair",
    "verify_v9_seal",
    "V10_SEALED_DIGESTS",
    "V11_ATTEMPT_ID",
    "V11_ATTEMPT_RECEIPT_REL",
    "V11_AUTH_REL",
    "V11_BLOCKED_REL",
    "V11_DRY_RUN_REL",
    "V11_LOGICAL_CALL_ID",
    "V11_MANIFEST_REL",
    "V11_POLICY_REL",
    "V11_PRIVATE_ROOT",
    "V11_READY_REL",
    "V11_RELEASE_REL",
    "V11_SESSION_ID",
    "V11_VERIFICATION_REL",
    "expected_prefreeze_v11_manifest",
    "expected_v11_authorization",
    "expected_v11_provider_release",
    "expected_v11_retry_policy",
    "prefreeze_v11_runtime_spec",
    "provider_free_v11_dry_run",
    "v11_physical_root",
    "validate_prefreeze_v11",
    "verify_v10_seal",
]
