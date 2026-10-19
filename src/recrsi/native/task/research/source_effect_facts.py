"""Static execution-footprint facts; never executes or rewrites candidate code.

Direct self.method edges are syntactic evidence, not runtime coverage. These
facts supplement the original source and measured TRAIN/DEV evidence. They do
not reject legal architectures, grade novelty, or infer recommendation quality.
"""
import ast
import hashlib


def source_facts(source):
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    cls = next((n for n in classes if n.name == "FreshCandidateModel"), None)
    if cls is None:
        raise ValueError("Select the actual candidate class explicitly before using this extractor")
    methods = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    edges, normalized, attributes, unresolved = {}, {}, {}, {}
    for name, node in methods.items():
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        direct = sorted({n.func.attr for n in calls if isinstance(n.func, ast.Attribute)
                         and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"})
        edges[name] = [m for m in direct if m in methods]
        unresolved[name] = [m for m in direct if m not in methods]
        attributes[name] = sorted({n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
                                   and isinstance(n.value, ast.Name) and n.value.id == "self"})
        normalized[name] = [dict(line=n.lineno, literal=ast.get_source_segment(source, n))
                            for n in calls if ast.unparse(n.func) in
                            ("F.normalize", "torch.nn.functional.normalize")]

    def reach(start):
        todo, found = [start] if start in methods else [], set()
        while todo:
            name = todo.pop()
            if name in found:
                continue
            found.add(name)
            todo.extend(edges[name])
        return sorted(found)

    roots = {n: reach(n) for n in ("calculate_loss", "predict", "full_sort_predict")}
    shared = sorted(set(roots["calculate_loss"]) &
                    (set(roots["predict"]) | set(roots["full_sort_predict"])))
    return dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                method_ast_sha256={n: hashlib.sha256(ast.dump(v, include_attributes=False).encode()).hexdigest()
                                   for n, v in methods.items()},
                direct_self_method_edges=edges, unresolved_self_calls=unresolved,
                self_attributes_by_method=attributes, syntactically_reachable=roots,
                shared_train_inference_methods=shared,
                normalize_call_sites=normalized,
                evidence_class="STATIC_SOURCE_FACTS_NOT_RUNTIME_OR_QUALITY",
                limits=["Branches, dynamic dispatch, inherited methods, aliases and callbacks require actual execution evidence.",
                        "Self attributes are source names, not proof of trainable parameter identity.",
                        "Normalization placement is not a causal performance effect; no candidate is rejected."])


def compare_sources(parent_source, candidate_source):
    parent, candidate = source_facts(parent_source), source_facts(candidate_source)
    names = sorted(set(parent["method_ast_sha256"]) | set(candidate["method_ast_sha256"]))
    changed = [n for n in names if parent["method_ast_sha256"].get(n) != candidate["method_ast_sha256"].get(n)]
    attribute_changes = {}
    for name in changed:
        before = set(parent["self_attributes_by_method"].get(name, []))
        after = set(candidate["self_attributes_by_method"].get(name, []))
        attribute_changes[name] = dict(removed=sorted(before-after), added=sorted(after-before))
    return dict(parent=parent, candidate=candidate, changed_methods=changed,
                changed_shared_train_inference_methods=sorted(set(changed) & set(candidate["shared_train_inference_methods"])),
                attribute_changes=attribute_changes,
                claim="Differences requiring explanation, not an automated equivalence or efficacy judgement.")
