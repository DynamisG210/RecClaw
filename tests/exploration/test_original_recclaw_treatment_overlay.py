from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_launcher import materialize_arm


def _method(tree: ast.AST, name: str) -> ast.FunctionDef:
    matched = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matched) != 1:
        raise AssertionError(f"expected exactly one method named {name}")
    return matched[0]


def _self_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and isinstance(item.func.value, ast.Name)
        and item.func.value.id == "self"
        and item.func.attr == name
    ]


def _nearest_if_test(node: ast.AST, target: ast.AST) -> str | None:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    current = target
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.If):
            return ast.unparse(current.test)
    return None


class OriginalRecClawTreatmentOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        runtime_root = Path(cls.temporary.name) / "runtime"
        record = materialize_arm(
            runtime_root=runtime_root,
            arm="treatment",
            search_seed=42,
        )
        cls.source_root = Path(record["source_root"])
        cls.source_text = (cls.source_root / "scripts" / "agent.py").read_text(
            encoding="utf-8"
        )
        cls.tree = ast.parse(cls.source_text)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_ordinary_raw_trial_and_frontier_are_guarded_before_write(self) -> None:
        run = _method(self.tree, "run")
        remember_calls = [
            call
            for call in _self_calls(run, "remember")
            if call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "record"
        ]
        self.assertEqual(1, len(remember_calls))
        self.assertEqual(
            "remember_original_trial",
            _nearest_if_test(run, remember_calls[0]),
        )
        frontier_calls = _self_calls(run, "_update_history_best")
        self.assertEqual(1, len(frontier_calls))
        self.assertEqual(
            "update_primary_frontier",
            _nearest_if_test(run, frontier_calls[0]),
        )
        postcheck_calls = [
            call
            for call in ast.walk(run)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "postcheck"
        ]
        self.assertEqual(1, len(postcheck_calls))
        self.assertLess(postcheck_calls[0].lineno, remember_calls[0].lineno)
        self.assertLess(postcheck_calls[0].lineno, frontier_calls[0].lineno)
        self.assertIn('remember_original_trial = True', ast.get_source_segment(self.source_text, run))
        self.assertIn('update_primary_frontier = True', ast.get_source_segment(self.source_text, run))
        self.assertIn(
            'if feedback_recorded and disposition == "QUARANTINE_ORIGINAL_TRIAL"',
            ast.get_source_segment(self.source_text, run),
        )

    def test_planner_seed_validation_is_guarded_before_original_event_write(self) -> None:
        method = _method(self.tree, "verify_last_keep")
        seed_events = []
        for call in _self_calls(method, "remember_event"):
            if not call.args or not isinstance(call.args[0], ast.Dict):
                continue
            event_dict = call.args[0]
            for key, value in zip(event_dict.keys, event_dict.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "event"
                    and isinstance(value, ast.Constant)
                    and value.value == "seed_validation"
                ):
                    seed_events.append(call)
        self.assertEqual(1, len(seed_events))
        self.assertEqual(
            "remember_original_validation",
            _nearest_if_test(method, seed_events[0]),
        )
        postchecks = [
            call
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "postcheck_seed_validation"
        ]
        self.assertEqual(1, len(postchecks))
        self.assertLess(postchecks[0].lineno, seed_events[0].lineno)
        self.assertIn(
            "if feedback_recorded:\n                    remember_original_validation",
            ast.get_source_segment(self.source_text, method),
        )

    def test_runtime_feedback_is_decorated_before_persistence_and_memory(self) -> None:
        method = _method(self.tree, "remember_guard_feedback")
        prepare = [
            call
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "prepare_feedback_for_original_memory"
        ]
        persist = [
            call
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "persist_feedback"
        ]
        appends = [
            call
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "append"
        ]
        self.assertEqual(1, len(prepare))
        self.assertEqual(1, len(persist))
        self.assertTrue(appends)
        self.assertLess(prepare[0].lineno, persist[0].lineno)
        self.assertTrue(all(prepare[0].lineno < call.lineno for call in appends))

    def test_feedback_persistence_failure_fails_open_in_all_treatment_paths(self) -> None:
        feedback_method = ast.get_source_segment(
            self.source_text, _method(self.tree, "remember_guard_feedback")
        )
        run_method = ast.get_source_segment(self.source_text, _method(self.tree, "run"))
        validation_method = ast.get_source_segment(
            self.source_text, _method(self.tree, "verify_last_keep")
        )
        self.assertIn("except Exception as exc", feedback_method)
        self.assertIn("return False", feedback_method)
        self.assertIn(
            "if feedback_recorded and self.guard_hook.should_defer(guard_precheck)",
            run_method,
        )
        self.assertIn(
            'if feedback_recorded and disposition == "QUARANTINE_ORIGINAL_TRIAL"',
            run_method,
        )
        self.assertIn(
            "if feedback_recorded:\n                    remember_original_validation",
            validation_method,
        )


if __name__ == "__main__":
    unittest.main()
