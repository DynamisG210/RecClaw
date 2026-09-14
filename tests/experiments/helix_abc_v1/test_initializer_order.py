"""Portable constructor-order regressions from the verified repair delivery."""
import ast
import unittest
from recclaw_core.experiments.helix_abc_v1 import conversion_efficiency as fixed

PATH = "recclaw_ext/candidate.py"


def scope(current, revised, method="helper"):
    return fixed.scope_mechanical_repair_source(
        current_source={PATH: current}, repaired_source={PATH: revised},
        failure={"stage": "CONSTRUCTION", "reason_code": "ATTRIBUTEERROR",
                 "implicated_methods": [method]},
    )[PATH]


def model(source):
    namespace = {}
    exec(compile(source, PATH, "exec"), namespace)
    return namespace["FreshCandidateModel"]()


SOURCE = '''class FreshCandidateModel:
    def __init__(self):
        count = 2
        self.scale = 7
        self.value = self.helper()
        self.state = {"count": count}
    def helper(self):
        self.state["count"] += 1
        return self.state["count"]
    def calculate_loss(self):
        return self.scale * self.value
    def predict(self):
        return self.value
'''


def repair(source):
    line = next(row for row in source.splitlines(True) if "self.state" in row and "count}" in row)
    return source.replace(line, "").replace("        self.value", line + "        self.value", 1)


class InitializerOrderTests(unittest.TestCase):
    def test_reorder_retains_unrelated_constructor_loss_and_score(self):
        revised = repair(SOURCE).replace("self.scale = 7", "self.scale = 99").replace("self.scale * self.value", "0").replace("return self.value", "return 99")
        with self.assertRaises(AttributeError):
            model(SOURCE)
        out = scope(SOURCE, revised)
        instance = model(out)
        self.assertEqual((instance.value, instance.calculate_loss(), instance.predict()), (3, 21, 3))
        self.assertEqual(ast.dump(ast.parse(out)), ast.dump(ast.parse(repair(SOURCE))))

    def test_renamed_annotated_state_and_indirect_constructor_call(self):
        for state in ("counts", "routing_state"):
            for annotated in (False, True):
                with self.subTest(state=state, annotated=annotated):
                    original = SOURCE.replace("self.state", f"self.{state}")
                    revised = repair(SOURCE).replace("self.state", f"self.{state}")
                    if annotated:
                        original = original.replace(f"self.{state} =", f"self.{state}: dict =")
                        revised = revised.replace(f"self.{state} =", f"self.{state}: dict =")
                    original = original.replace("self.value = self.helper()", "self.value = self.wrapper()")
                    revised = revised.replace("self.value = self.helper()", "self.value = self.wrapper()")
                    wrapper = "    def wrapper(self):\n        return self.helper()\n"
                    instance = model(scope(original + wrapper, revised + wrapper))
                    self.assertEqual(instance.calculate_loss(), 21)

    def test_healthy_constructor_order_is_unchanged(self):
        healthy = repair(SOURCE)
        revised = healthy.replace('+= 1', '+= 2').replace("self.scale = 7", "self.scale = 99")
        out = scope(healthy, revised)
        get_init = lambda text: ast.dump(next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "__init__"))
        self.assertEqual(get_init(out), get_init(healthy))
        self.assertEqual(model(out).calculate_loss(), 28)

    def test_missing_or_changed_local_prerequisite_is_not_silently_moved(self):
        for original in (SOURCE.replace("        count = 2\n", "").replace("        self.state", "        count = 2\n        self.state", 1), SOURCE):
            revised = repair(SOURCE).replace("count = 2", "count = 8").replace("+= 1", "+= 2")
            out = scope(original, revised)
            get_init = lambda text: ast.dump(next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "__init__"))
            self.assertEqual(get_init(out), get_init(original))
