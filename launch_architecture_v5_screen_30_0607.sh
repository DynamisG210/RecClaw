#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/tingrangan/projects/RecClaw_architecture_v5_screen_20260607
RUN_NAME="${RUN_NAME:-20260607_architecture_v5_screen_30_openai_searchquality}"
RUN_DIR="$PROJECT/results/pilots/$RUN_NAME"

cd "$PROJECT"
source /home/tingrangan/.recclaw_4router_env
mkdir -p "$RUN_DIR" "$RUN_DIR/candidates" "$RUN_DIR/overrides" "$RUN_DIR/smoke/candidates" "$RUN_DIR/smoke/overrides" "$RUN_DIR/checkpoints"

export PYTHONUNBUFFERED=1
export RECCLAW_RESULT_DIR="$RUN_DIR/candidates"
export RECCLAW_RESULTS_CSV="$RUN_DIR/results.csv"
export RECCLAW_OVERRIDE_DIR="$RUN_DIR/overrides"
export RECCLAW_CHECKPOINT_DIR="$RUN_DIR/checkpoints"
export RECCLAW_CHECKPOINT_POLICY=cleanup_all
export RECCLAW_SMOKE_RESULTS_CSV="$RUN_DIR/results.smoke.csv"
export RECCLAW_SMOKE_RESULT_DIR="$RUN_DIR/smoke/candidates"
export RECCLAW_SMOKE_OVERRIDE_DIR="$RUN_DIR/smoke/overrides"

DIRECTIVE="RecClaw v5 architecture search-quality screen. Keep the ML-1M general-recommendation full-sort protocol unchanged. The goal is to validate search behavior, not maximize budget: all formal rows must be genuine Custom architecture candidates, never BPR/LightGCN parent formal runs. Use architecture-native families: graph-attention propagation, bipartite rewiring/shortcut propagation, low-rank or router interaction head, clean residual propagation, contrastive or rank-aware objective head, hybrid graph-interaction, and transformer-style attention/FFN/residual/layer-norm blocks under the current general-rec protocol. Every proposal must state structural novelty, how it differs from recent failed signatures, and whether it is a repair. Repairs are allowed only for explicit implementation bugs with a novel or promising original proposal; do not repair low-scoring siblings by default. Smoke first; low-quality or collapsed smoke should be rejected before formal budget. Do not introduce SASRec/BERT4Rec, sequence dataloaders, temporal split, multimodal dependencies, RecBole core changes, data split changes, or LabLog runtime input. Prefer family coverage and clean audit trails over near-duplicate residual-norm/edge-drop interaction-head variants."

exec /home/tingrangan/venvs/recbole/bin/python scripts/agent.py \
  --rounds 30 \
  --start-round 1 \
  --loop-mode auto \
  --enable-candidate-proposals \
  --proposal-source llm \
  --proposal-mode architecture_first \
  --proposal-count 3 \
  --proposal-every 1 \
  --proposal-bonus 0.6 \
  --auto-promote-needs-review \
  --max-pending-implemented 6 \
  --max-implement-per-round 2 \
  --search-intensity balanced \
  --algorithm-budget-per-window 0 \
  --algorithm-first-explore-rounds 0 \
  --architecture-smoke-min-best-score 0.16 \
  --objective-smoke-min-best-score 0.16 \
  --smoke-quality-min-unique-scores 2 \
  --memory-path "$RUN_DIR/agent_memory.jsonl" \
  --state-summary-path "$RUN_DIR/agent_state_summary.json" \
  --proposal-path "$RUN_DIR/candidate_proposals.jsonl" \
  --proposal-history-path "$RUN_DIR/candidate_proposal_history.jsonl" \
  --results-csv "$RUN_DIR/results.csv" \
  --baseline-dir "$PROJECT/results/baseline" \
  --candidate-tree-path "$RUN_DIR/candidate_search_tree.json" \
  --candidate-tree-md-path "$RUN_DIR/candidate_search_tree.md" \
  --candidate-tree-mmd-path "$RUN_DIR/candidate_search_tree.mmd" \
  --experience-summary-path "$RUN_DIR/experience_summary.md" \
  --experience-summary-json-path "$RUN_DIR/experience_summary.json" \
  --reflection-memory-path "$RUN_DIR/reflection_memory.jsonl" \
  --refresh-experience-every 5 \
  --experiment-directive "$DIRECTIVE" \
  --checkpoint-dir "$RUN_DIR/checkpoints" \
  --checkpoint-policy cleanup_all \
  --set train_batch_size=2048 \
  --set eval_batch_size=65536 \
  --set worker=8 \
  --set gpu_id=1 \
  --llm-provider openai \
  --llm-api-key-env OPENAI_API_KEY \
  --llm-timeout 300 \
  --llm-max-tokens 7000 \
  --llm-retries 2 \
  >> "$RUN_DIR/agent.log" 2>&1
