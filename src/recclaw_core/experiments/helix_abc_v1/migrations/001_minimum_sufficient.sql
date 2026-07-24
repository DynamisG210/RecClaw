PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS scheduled_slots (
    experiment_id TEXT NOT NULL,
    arm_instance_id TEXT NOT NULL,
    arm_code TEXT NOT NULL CHECK (arm_code IN ('A', 'B', 'C')),
    search_seed INTEGER NOT NULL,
    round_index INTEGER NOT NULL CHECK (round_index >= 1),
    slot_status TEXT NOT NULL CHECK (
        slot_status IN ('PLANNED', 'OPENED', 'CLOSED', 'ABORTED', 'NOT_STARTED_STOP')
    ),
    round_id TEXT UNIQUE,
    stop_reason TEXT,
    PRIMARY KEY (experiment_id, arm_instance_id, search_seed, round_index)
);

CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    arm_instance_id TEXT NOT NULL,
    arm_code TEXT NOT NULL CHECK (arm_code IN ('A', 'B', 'C')),
    search_seed INTEGER NOT NULL,
    round_index INTEGER NOT NULL CHECK (round_index >= 1),
    idempotency_key TEXT NOT NULL UNIQUE,
    open_payload_digest TEXT NOT NULL CHECK (length(open_payload_digest) = 64),
    budget_snapshot_json TEXT NOT NULL,
    budget_snapshot_digest TEXT NOT NULL CHECK (length(budget_snapshot_digest) = 64),
    controller_state_before_digest TEXT NOT NULL CHECK (
        length(controller_state_before_digest) = 64
    ),
    controller_state_after_digest TEXT,
    status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED', 'ABORTED')),
    terminal_class TEXT,
    feedback_digest TEXT,
    FOREIGN KEY (experiment_id, arm_instance_id, search_seed, round_index)
        REFERENCES scheduled_slots (
            experiment_id, arm_instance_id, search_seed, round_index
        )
);

CREATE TABLE IF NOT EXISTS round_events (
    round_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL CHECK (event_seq >= 1),
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'ROUND_OPENED',
            'EXECUTION_CLAIMED',
            'EXECUTION_START_AMBIGUOUS',
            'ARTIFACT_REGISTERED',
            'ROUND_FEEDBACK',
            'ROUND_CLOSED',
            'RECOVERY_CLASSIFIED'
        )
    ),
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL CHECK (length(payload_digest) = 64),
    PRIMARY KEY (round_id, event_seq),
    FOREIGN KEY (round_id) REFERENCES rounds (round_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_round_feedback_per_round
    ON round_events (round_id)
    WHERE event_type = 'ROUND_FEEDBACK';

CREATE TABLE IF NOT EXISTS arm_state (
    experiment_id TEXT NOT NULL,
    arm_instance_id TEXT NOT NULL,
    arm_code TEXT NOT NULL CHECK (arm_code IN ('A', 'B', 'C')),
    search_seed INTEGER NOT NULL,
    experiment_contract_digest TEXT NOT NULL CHECK (
        length(experiment_contract_digest) = 64
    ),
    state TEXT NOT NULL CHECK (
        state IN ('ACTIVE', 'STOPPING', 'STOPPED', 'INCOMPLETE')
    ),
    next_round_index INTEGER NOT NULL CHECK (next_round_index >= 1),
    controller_state_digest TEXT NOT NULL CHECK (length(controller_state_digest) = 64),
    meta_policy_digest TEXT,
    search_memory_digest TEXT,
    stop_reason TEXT,
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    PRIMARY KEY (experiment_id, arm_instance_id, search_seed)
);

CREATE TABLE IF NOT EXISTS resource_ledger (
    ledger_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL,
    dimension TEXT NOT NULL CHECK (
        dimension IN (
            'PROPOSAL_GENERATION_SESSION',
            'PHYSICAL_LLM_CALL',
            'INPUT_TOKEN',
            'OUTPUT_TOKEN',
            'BILLED_TOKEN_DEBIT',
            'PROPOSAL',
            'WALL_TIME_MS',
            'RETRY',
            'PROPOSAL_ATTEMPT',
            'ORDINARY_EXECUTION',
            'COMMON_VALIDATION',
            'GPU_DEVICE_TIME_MS',
            'GPU_COST_MICROUNITS'
        )
    ),
    quantity INTEGER NOT NULL CHECK (quantity >= 0),
    unit TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_digest TEXT NOT NULL CHECK (length(payload_digest) = 64),
    FOREIGN KEY (round_id) REFERENCES rounds (round_id)
);

CREATE TABLE IF NOT EXISTS execution_claims (
    round_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL UNIQUE,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_digest TEXT NOT NULL CHECK (length(payload_digest) = 64),
    permit_digest TEXT NOT NULL CHECK (length(permit_digest) = 64),
    binding_digest TEXT NOT NULL CHECK (length(binding_digest) = 64),
    claim_state TEXT NOT NULL CHECK (
        claim_state IN (
            'CLAIMED',
            'STARTED',
            'START_AMBIGUOUS',
            'FINISHED',
            'NOT_STARTED'
        )
    ),
    execution_debited INTEGER NOT NULL DEFAULT 0 CHECK (
        execution_debited IN (0, 1)
    ),
    FOREIGN KEY (round_id) REFERENCES rounds (round_id)
);

CREATE TABLE IF NOT EXISTS triplet_barrier (
    experiment_id TEXT NOT NULL,
    search_seed INTEGER NOT NULL,
    round_index INTEGER NOT NULL CHECK (round_index >= 1),
    arm_a_instance_id TEXT NOT NULL,
    arm_b_instance_id TEXT NOT NULL,
    arm_c_instance_id TEXT NOT NULL,
    closed_bitmap INTEGER NOT NULL DEFAULT 0 CHECK (
        closed_bitmap BETWEEN 0 AND 7
    ),
    stop_requested INTEGER NOT NULL DEFAULT 0 CHECK (stop_requested IN (0, 1)),
    next_index_authorized INTEGER NOT NULL DEFAULT 0 CHECK (
        next_index_authorized IN (0, 1)
    ),
    stop_reason TEXT,
    stop_idempotency_key TEXT UNIQUE,
    stop_payload_digest TEXT,
    recovery_idempotency_key TEXT UNIQUE,
    recovery_payload_digest TEXT,
    recovery_result_json TEXT,
    PRIMARY KEY (experiment_id, search_seed, round_index),
    CHECK (
        next_index_authorized = 0
        OR (closed_bitmap = 7 AND stop_requested = 0)
    )
);

CREATE TABLE IF NOT EXISTS artifact_index (
    artifact_id TEXT PRIMARY KEY,
    round_id TEXT,
    artifact_type TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    producer TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_digest TEXT NOT NULL CHECK (length(payload_digest) = 64),
    UNIQUE (round_id, relative_path),
    FOREIGN KEY (round_id) REFERENCES rounds (round_id)
);

CREATE TRIGGER IF NOT EXISTS round_events_no_update
BEFORE UPDATE ON round_events
BEGIN
    SELECT RAISE(ABORT, 'round_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS round_events_no_delete
BEFORE DELETE ON round_events
BEGIN
    SELECT RAISE(ABORT, 'round_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS resource_ledger_no_update
BEFORE UPDATE ON resource_ledger
BEGIN
    SELECT RAISE(ABORT, 'resource_ledger is append-only');
END;

CREATE TRIGGER IF NOT EXISTS resource_ledger_no_delete
BEFORE DELETE ON resource_ledger
BEGIN
    SELECT RAISE(ABORT, 'resource_ledger is append-only');
END;

CREATE TRIGGER IF NOT EXISTS artifact_index_no_update
BEFORE UPDATE ON artifact_index
BEGIN
    SELECT RAISE(ABORT, 'artifact_index is append-only');
END;

CREATE TRIGGER IF NOT EXISTS artifact_index_no_delete
BEFORE DELETE ON artifact_index
BEGIN
    SELECT RAISE(ABORT, 'artifact_index is append-only');
END;

PRAGMA user_version = 1;
