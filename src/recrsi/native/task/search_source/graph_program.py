"""Small graph source/metric binding for native consumers; no search policy."""
import ast
import math


def program_files(program):
    """A complete single-file model replaces the MoRec five-file carrier."""
    tree = ast.parse(program)
    if not any(isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel" for node in tree.body):
        raise ValueError("Graph candidate must include its complete FreshCandidateModel")
    return {"candidate.py": program}


def primary_score(result, binding):
    from graph_metric_contract import require_binding
    require_binding(binding)
    if result.get("primary_metric") != "NDCG@5":
        raise ValueError("A checkpoint selected by NDCG@5 is required")
    if result.get("status") != "SUCCESS" or result.get("test_accessed") is not False:
        raise ValueError("No successful validation-only graph result")
    if result.get("task_id") != binding["task_id"]:
        raise ValueError("Graph task identities must not be mixed")
    key = binding["evaluation"]["primary_metric"].lower()
    score = result["metrics"][key]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid primary validation metric")
    return score


def common_task_text(binding):
    if binding.get('adapter') == 'recbole-sequential-fixed100-v1':
        return (
            f"Improve the supplied original {binding['line']} sequential recommender on {binding['dataset']}. "
            "The complete candidate.py exposes FreshCandidateModel(config, train_dataset), "
            "calculate_loss(Interaction), predict and full_sort_predict. Use RecBole SequentialRecommender "
            "with item_id_list and item_length; chronological prefixes predict the next item with CE loss. "
            "Only TRAIN prefix/target examples enter the model constructor and optimizer. "
            "At validation, score the supplied prefix, never the target or future events. "
            "The trusted worker owns ID mapping, Adam, early stopping, checkpoint selection, "
            "fixed one-positive plus99-negative ranking and deterministic item-ID tie breaking. "
            "Do not access DEV labels or held-out TEST from candidate code. "
            f"Primary metric: {binding['evaluation']['primary_metric']}. "
            "Preserve native mechanism generation, source selection and feedback; do not import old winners."
        )
    return (
        f"Improve the supplied original {binding['line']} graph recommender on {binding['dataset']}. "
        "All six systems receive the same original model code and these raw task facts. "
        "The complete candidate.py exposes FreshCandidateModel(config, train_dataset), "
        "calculate_loss(pairwise Interaction), predict and full_sort_predict. "
        "Graph construction uses only train interactions; candidate.py owns model computation and loss. "
        "The trusted worker owns ID mapping, uniform train-only negative sampling, Adam, "
        "training/early stopping, fixed 1-positive plus 99-negative evaluation and checkpoint selection. "
        "Do not read dev labels in model code or access heldout data. The fixed candidate pool excludes padding0 "
        "and seen positives. Rankings use deterministic item-ID tie breaking. Inference caches must be cleared before training. "
        f"Primary metric: {binding['evaluation']['primary_metric']}. "
        "Preserve each research system's native generation, feedback and selection; "
        "do not import any other system's generated mechanisms or old winners."
    )
