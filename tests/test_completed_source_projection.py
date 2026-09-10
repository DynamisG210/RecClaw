"""Small public fixtures for completed-source presentation, without API calls."""
from copy import deepcopy
from difflib import unified_diff

from recclaw_core.research_line.interfaces import project_provider_context_view


def context(parent, completed):
    return {
        "scientific_memory": {"global_memory": {}},
        "lineage_parent_mechanism_program": {
            "candidate_id": "parent", "program_digest": "parent-program",
            "mechanism_program": {},
            "source_bundle": {
                "candidate_id": "parent", "source_tree_digest": "parent-tree",
                "files": [{"path": "candidate.py", "content": parent}],
            },
        },
        "latest_completed_execution": {
            "metric": 0.1, "elapsed_seconds": 123,
            "implementation": {
                "source_tree_digest": "completed-tree",
                "files": [{"path": "candidate.py", "content": completed}],
            },
        },
    }


def test_changed_completed_file_uses_exact_parent_and_standard_diff():
    parent = "".join(f"value_{i} = {i}\n" for i in range(100))
    completed = parent.replace("value_50 = 50\n", "value_50 = 51\n")
    raw = context(parent, completed)
    before = deepcopy(raw)
    result = project_provider_context_view(raw)["state"]
    row = result["latest_completed_execution"]["implementation"]["files"][0]
    assert raw == before
    assert result["parent_implementation"]["files"][0]["content"] == parent
    assert row["source_context_ref"] == "state.parent_implementation.files[0]"
    assert row["content_diff"] == "".join(unified_diff(
        parent.splitlines(keepends=True), completed.splitlines(keepends=True),
        fromfile="parent/candidate.py", tofile="completed/candidate.py",
    ))
    assert "content" not in row
    assert result["latest_completed_execution"]["metric"] == 0.1
    assert result["latest_completed_execution"]["elapsed_seconds"] == 123


def test_missing_final_newline_and_large_diff_remain_full_text():
    for completed in ("no final newline", "entirely different\n"):
        raw = context("old\n", completed)
        result = project_provider_context_view(raw)["state"]["latest_completed_execution"]
        assert result["implementation"]["files"][0]["content"] == completed


def test_same_tree_reference_remains_unchanged():
    raw = context("same\n", "same\n")
    raw["latest_completed_execution"]["implementation"]["source_tree_digest"] = "parent-tree"
    result = project_provider_context_view(raw)["state"]["latest_completed_execution"]
    assert result["implementation"] == {
        "source_tree_digest": "parent-tree", "source_context_ref": "state.parent_implementation",
    }
