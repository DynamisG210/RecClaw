"""Bind a declared local BL-ICF score to the retained parent computation.

The research actor supplies the score, not new embeddings, auxiliary losses or
optimizer defaults. Native/non-BPR objectives never select this BPR hook.
"""

from __future__ import annotations

import ast
import copy
from typing import Callable


def _row_dimension(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant) and node.value == 1
        or isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant) and node.operand.value == 1
    )


class _ScoreCalls(ast.NodeTransformer):
    """Replace the parent's row-dot or matrix-dot at a scoring seam only."""

    def __init__(self) -> None:
        self.bound = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "recclaw_score_head":
            self.bound += 1
            return node
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "sum"
            and isinstance(node.func.value, ast.BinOp)
            and isinstance(node.func.value.op, ast.Mult)
            and (
                (len(node.args) == 1 and _row_dimension(node.args[0]))
                or any(key.arg == "dim" and _row_dimension(key.value) for key in node.keywords)
            )
        ):
            product = node.func.value
            return self.score(product.left, product.right, pairwise=True, location=node)
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if isinstance(node.op, ast.MatMult):
            right = node.right
            if isinstance(right, ast.Call) and isinstance(right.func, ast.Attribute) and right.func.attr in {"transpose", "t"}:
                return self.score(node.left, right.func.value, pairwise=False, location=node)
            if isinstance(right, ast.Attribute) and right.attr == "T":
                return self.score(node.left, right.value, pairwise=False, location=node)
        return self.generic_visit(node)

    def score(self, users: ast.AST, items: ast.AST, *, pairwise: bool, location: ast.AST) -> ast.Call:
        self.bound += 1
        return ast.copy_location(ast.Call(
            func=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="recclaw_score_head", ctx=ast.Load()),
            args=[copy.deepcopy(users), copy.deepcopy(items)],
            keywords=[ast.keyword(arg="pairwise", value=ast.Constant(value=pairwise))],
        ), location)


def bind_parent_score(
    tree: ast.Module,
    entrypoint: ast.ClassDef,
    *,
    resolve_method: Callable[[str], ast.FunctionDef | None],
    train_objective: bool,
) -> None:
    # Only a complete plain-dot/hook score is a local replacement seam.
    # Replacing a dot inside old calibration would silently retain that head.
    for name in ("predict", "full_sort_predict"):
        method = resolve_method(name)
        if method is None or not _complete_score_seam(method):
            raise ValueError(f"parent {name} requires full score implementation")

    def install(method: ast.FunctionDef) -> None:
        entrypoint.body = [node for node in entrypoint.body if not (
            isinstance(node, ast.FunctionDef) and node.name == method.name
        )]
        entrypoint.body.append(method)

    for name in ("predict", "full_sort_predict"):
        parent_method = resolve_method(name)
        if parent_method is None:
            raise ValueError(f"parent score scaffold lacks {name}")
        transform = _ScoreCalls()
        method = transform.visit(copy.deepcopy(parent_method))
        if transform.bound != 1:
            raise ValueError(f"parent {name} lacks one exact score seam")
        install(method)
    if not train_objective:
        return
    loss = resolve_method("calculate_loss")
    if loss is None:
        raise ValueError("parent score scaffold lacks calculate_loss")
    loss = copy.deepcopy(loss)
    calls = [node for node in ast.walk(loss) if isinstance(node, ast.Call)]
    if any(isinstance(node.func, ast.Attribute) and node.func.attr == "recclaw_score_objective" for node in calls):
        # A later score variant changes only its hook; inherited wiring remains.
        return
    primary = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "_bpr_loss"]
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_bpr_loss"]
    if not primary:
        # Standard inline BPR parents expose the same two score seams. Keep
        # their native loss callable/formula and auxiliary terms untouched.
        transform = _ScoreCalls()
        loss = transform.visit(loss)
        if transform.bound == 2:
            install(loss)
            return
    if len(primary) != 1 or len(helpers) != 1:
        raise ValueError("declared BPR score hook requires the parent's exact _bpr_loss seam")
    helper = copy.deepcopy(helpers[0])
    helper.name = "recclaw_score_objective"
    helper.args.args.insert(0, ast.arg(arg="self"))
    transform = _ScoreCalls()
    helper = transform.visit(helper)
    if transform.bound != 2:
        raise ValueError("parent BPR helper must expose positive and negative score seams")
    primary[0].func = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=helper.name, ctx=ast.Load())
    install(helper)
    install(loss)


def _complete_score_seam(method: ast.FunctionDef) -> bool:
    """Recognize a returned score, allowing aliases and output reshaping only."""
    returns = [node for node in ast.walk(method) if isinstance(node, ast.Return)]
    if len(returns) != 1:
        return False
    assignments = {
        node.targets[0].id: node.value
        for node in method.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    value = returns[0].value
    seen: set[str] = set()
    while True:
        if isinstance(value, ast.Name) and value.id in assignments:
            if value.id in seen:
                return False
            seen.add(value.id)
            value = assignments[value.id]
        elif (
            isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr in {"view", "reshape", "flatten", "contiguous"}
        ):
            value = value.func.value
        else:
            break
    # Only the returned alias/reshape chain owns this score. An extra alias,
    # mutation, or helper call can retain part of the old head; use the existing
    # full-source path for that parent rather than guessing Python alias effects.
    for statement in method.body:
        if statement is returns[0]:
            continue
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in seen):
            continue
        if any(isinstance(node, ast.Name) and node.id in seen
               for node in ast.walk(statement)):
            return False
    transform = _ScoreCalls()
    score = transform.visit(copy.deepcopy(value)) if value is not None else None
    return bool(
        transform.bound == 1 and isinstance(score, ast.Call)
        and isinstance(score.func, ast.Attribute)
        and score.func.attr == "recclaw_score_head"
    )


def parent_score_is_bindable(source: str, *, train_objective: bool) -> bool:
    """Select local versus existing full-source implementation before the call."""
    try:
        tree = ast.parse(source)
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        entrypoint = classes["FreshCandidateModel"]

        def resolve(name: str, cls: ast.ClassDef = entrypoint, seen: frozenset[str] = frozenset()) -> ast.FunctionDef | None:
            if cls.name in seen:
                return None
            direct = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name]
            if direct:
                return direct[0] if len(direct) == 1 else None
            inherited = [
                method for base in cls.bases
                if isinstance(base, ast.Name) and base.id in classes
                for method in (resolve(name, classes[base.id], seen | {cls.name}),)
                if method is not None
            ]
            return inherited[0] if len(inherited) == 1 else None

        bind_parent_score(tree, entrypoint, resolve_method=resolve, train_objective=train_objective)
    except (SyntaxError, KeyError, ValueError):
        return False
    return True
