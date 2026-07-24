ALTER TABLE execution_claims
    ADD COLUMN runtime_release_digest TEXT
        CHECK (runtime_release_digest IS NULL OR length(runtime_release_digest) = 64);
ALTER TABLE execution_claims
    ADD COLUMN experiment_id TEXT;
ALTER TABLE execution_claims
    ADD COLUMN candidate_id TEXT;
ALTER TABLE execution_claims
    ADD COLUMN run_id TEXT;
ALTER TABLE execution_claims
    ADD COLUMN budget_digest TEXT
        CHECK (budget_digest IS NULL OR length(budget_digest) = 64);
ALTER TABLE execution_claims
    ADD COLUMN runtime_binding_digest TEXT
        CHECK (runtime_binding_digest IS NULL OR length(runtime_binding_digest) = 64);
ALTER TABLE execution_claims
    ADD COLUMN runner_abi TEXT;
ALTER TABLE execution_claims
    ADD COLUMN execution_purpose TEXT;
ALTER TABLE execution_claims
    ADD COLUMN metric_contract_digest TEXT
        CHECK (metric_contract_digest IS NULL OR length(metric_contract_digest) = 64);
ALTER TABLE execution_claims
    ADD COLUMN resource_contract_digest TEXT
        CHECK (resource_contract_digest IS NULL OR length(resource_contract_digest) = 64);
ALTER TABLE execution_claims
    ADD COLUMN attempt_state TEXT NOT NULL DEFAULT 'NOT_PREPARED'
        CHECK (attempt_state IN ('NOT_PREPARED', 'PREPARED', 'START_CONFIRMED'));
ALTER TABLE execution_claims
    ADD COLUMN ordinary_launch_attempt_ordinal INTEGER
        CHECK (
            ordinary_launch_attempt_ordinal IS NULL
            OR ordinary_launch_attempt_ordinal = 1
        );

PRAGMA user_version = 2;
