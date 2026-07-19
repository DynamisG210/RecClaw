# Trustworthy-evidence line

The development Evidence Guard accepts five explicit inputs:

1. Claim;
2. Protocol;
3. Current Evidence;
4. Action Proposal;
5. optional Observation.

It returns development action legality, protocol diagnostics, evidence disposition, affected claim scope, claim ceiling, and bounded next-iteration feedback. The schemas are under `schemas/` and the frozen implementation is under `src/recclaw_core/exploration/`.

The integration preserves four separations:

- execution permission is not evidence authority;
- a runner success is not automatically admissible evidence;
- search memory is not accepted evidence history;
- development claim ceilings are not authoritative ClaimRecord transitions.

Pre-run behavior may request revision or a separate protocol branch. A Guard error or persistence failure fails open to the Original RecClaw decision. Post-run behavior admits only valid improvements and informative negatives to primary research memory. Runtime blockers remain diagnostic signals; protocol mismatches and incomplete provenance are retained in the audit log but quarantined from the current claim and primary frontier.

The outward experiment vocabulary uses `PROTOCOL_COMPATIBLE_FOR_DEVELOPMENT`; the frozen core's older internal `EXECUTE_DEVELOPMENT_ONLY` token is not interpreted as execution authorization.

Supported profile families remain `OFFLINE_TOPN` and `SEQUENTIAL_NEXT_ITEM`. Phase 1 uses only the fixed ML-1M `OFFLINE_TOPN` random-user split and full-sort evaluation contract. No OPE, online evaluation, new profile family, permission release, formal claim acceptance, or production promotion is included.
