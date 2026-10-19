"""Replace the primary term in the native BL parent loss, retaining its tail.

The supported source seam is deliberately structural: interaction indexing,
one forward tuple, a primary computation, then independent auxiliary assignments
and a loss tuple. Other loss programs keep the full-source implementation path.
No objective formula is recognized or rewritten here.
"""

from __future__ import annotations

import ast
import copy
from typing import Any, Callable, Mapping


PRIMARY_METHOD = "recclaw_primary_objective"
PRIMARY_INITIALIZER = "recclaw_initialize_primary_objective"


def retained_components_match(
    source: str, current: Mapping[str, Any], *, require_metadata: bool,
) -> bool:
    """Use the exact parent's existing compiler literals, not slot occupancy.

    Old root sources predate those literals. They may use the original declared
    local-delta contract, but cannot establish a nonlocal replacement history.
    """
    try:
        assignment = next((node for node in ast.parse(source).body if (
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "RECCLAW_IMPLEMENTED_COMPONENT_SPECS"
                for target in node.targets
            )
        )), None)
        if assignment is None:
            return not require_metadata
        parent = ast.literal_eval(assignment.value)
        if not isinstance(parent, dict):
            return False
        for name in parent.keys() | current.keys():
            before, after = parent.get(name), current.get(name)
            if before == after:
                continue
            if any(value is not None and (
                not isinstance(value, Mapping) or value.get("slot_id") != "PRIMARY_OBJECTIVE"
            ) for value in (before, after)):
                return False
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


def _self_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == name
    )


def _primary_loss(loss: ast.FunctionDef) -> ast.FunctionDef:
    loss = copy.deepcopy(loss)
    result = loss.body[-1]
    if not (
        isinstance(result, ast.Return)
        and isinstance(result.value, ast.Tuple)
        and len(result.value.elts) in (2, 3)
        and all(isinstance(term, ast.Name) for term in result.value.elts)
        and sum(isinstance(node, ast.Return) for node in ast.walk(loss)) == 1
    ):
        raise ValueError("parent loss does not expose the native BL loss tuple")
    terms = result.value.elts
    tail_start = len(loss.body) - len(terms)
    auxiliary = loss.body[tail_start:-1]
    if any(
        not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or ast.dump(statement.targets[0]) != ast.dump(ast.Name(id=term.id, ctx=ast.Store()))
        for statement, term in zip(auxiliary, terms[1:])
    ):
        raise ValueError("parent auxiliary terms are not independent final assignments")
    forwards = [
        (index, statement)
        for index, statement in enumerate(loss.body[:tail_start])
        if isinstance(statement, ast.Assign) and _self_call(statement.value, "forward")
    ]
    if len(forwards) != 1:
        raise ValueError("parent loss lacks one native forward tuple")
    forward_index, forward = forwards[0]
    if not (
        len(forward.targets) == 1
        and isinstance(forward.targets[0], ast.Tuple)
        and len(forward.targets[0].elts) in (2, 4)
        and all(isinstance(item, ast.Name) for item in forward.targets[0].elts)
    ):
        raise ValueError("parent forward does not expose user/item representations")
    for statement in loss.body[:forward_index]:
        indexing = (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Subscript)
            and isinstance(statement.value.value, ast.Name)
            and statement.value.value.id == "interaction"
        )
        cache_clear = isinstance(statement, ast.Expr) and _self_call(statement.value, "_clear_restore")
        if not (indexing or cache_clear):
            raise ValueError("parent loss setup requires complete implementation")
    removed_names = {
        node.id for statement in loss.body[forward_index + 1:tail_start]
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in removed_names
        for statement in auxiliary for node in ast.walk(statement)
    ):
        raise ValueError("parent auxiliary losses depend on the replaced primary computation")
    primary = ast.Assign(
        targets=[ast.Name(id=terms[0].id, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=PRIMARY_METHOD, ctx=ast.Load()),
            args=[ast.Name(id="interaction", ctx=ast.Load()), ast.Tuple(
                elts=[ast.Name(id=item.id, ctx=ast.Load()) for item in forward.targets[0].elts],
                ctx=ast.Load(),
            )], keywords=[],
        ),
    )
    loss.body = loss.body[:forward_index + 1] + [primary] + auxiliary + [result]
    return loss


def bind_parent_objective(
    entrypoint: ast.ClassDef,
    *,
    resolve_method: Callable[[str], ast.FunctionDef | None],
) -> None:
    loss, initializer = resolve_method("calculate_loss"), resolve_method("__init__")
    if loss is None or initializer is None:
        raise ValueError("parent lacks an explicit loss or constructor")
    loss = _primary_loss(loss)
    direct_initializer = next((node for node in entrypoint.body if (
        isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )), None)
    # Copying an inherited __init__ into the child would change zero-argument
    # super() and run parent graph construction twice. Delegate to it once.
    initializer = copy.deepcopy(direct_initializer) if direct_initializer is not None else ast.parse(
        "def __init__(self, config, dataset):\n    super().__init__(config, dataset)\n"
    ).body[0]
    if not any(_self_call(node, PRIMARY_INITIALIZER) for node in ast.walk(initializer)):
        initializer.body.extend(ast.parse(
            f"self.{PRIMARY_INITIALIZER}(config, dataset)"
        ).body)
    entrypoint.body = [node for node in entrypoint.body if not (
        isinstance(node, ast.FunctionDef) and node.name in {"calculate_loss", "__init__"}
    )] + [loss, initializer]


def parent_objective_is_bindable(source: str) -> bool:
    try:
        tree = ast.parse(source)
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        entrypoint = classes["FreshCandidateModel"]

        def resolve(name: str, cls: ast.ClassDef = entrypoint) -> ast.FunctionDef | None:
            direct = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name]
            if direct:
                return direct[0] if len(direct) == 1 else None
            bases = [base for base in cls.bases if isinstance(base, ast.Name) and base.id in classes]
            return resolve(name, classes[bases[0].id]) if len(bases) == 1 else None

        bind_parent_objective(entrypoint, resolve_method=resolve)
    except (SyntaxError, KeyError, ValueError, RecursionError):
        return False
    return True
