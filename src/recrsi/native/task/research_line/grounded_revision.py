"""Two paid passes by the same Research role; native selection and execution stay intact."""
import ast
import copy
import difflib
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from program_adapter import ProgramAdapter, program_scope, program_projection

VERSION = 'ngcf-independent-discovery-v1'
STAGES = ('draft', 'review')
REVIEW = (
    'This is the second and final pass on your own unexecuted draft, not a new candidate '
    'opportunity. Return one final complete proposal using the same schema. Reconsider '
    'the actual computation, parent-relative changes, evidence and falsifiable prediction. '
    'Use the existing narrative fields to state what you retained or revised and why. '
    'Distinguish a measured result from your interpretation; cite relevant own-run '
    'candidate IDs and numerical outcomes, including negative results, and explain '
    'what they change about this proposal. No measured predecessor means no empirical '
    'learning claim. You may retain the draft, simplify it, or make a coherent structural '
    'revision; no mechanism family is preferred. Do not invent experiments. Only this '
    'final proposal may enter the original selection and execution path.'
)
GROUNDING = (
    'The attached code comparison is generated from the exact supplied root, declared '
    'construction parent and your draft. It is static source evidence, not an execution, '
    'mathematical proof or model-quality score. Reconcile your explanation against both '
    'comparisons, including initialization, propagation, readout, scoring and loss where '
    'they actually change. Operator trees show Python grouping explicitly. Identify '
    'where an explanation conflicts with code or overstates what a negative result '
    'isolates; revise the computation and explanation together when justified. If you '
    'claim that disabling a change restores a reference, identify the precise reference '
    'and remaining changed functions. Unknown dynamic behavior remains unknown.'
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf8')).hexdigest()


def source_hash(source):
    return hashlib.sha256(source.encode('utf8')).hexdigest()


def operator_tree(node):
    if isinstance(node, ast.BinOp):
        return [type(node.op).__name__, operator_tree(node.left), operator_tree(node.right)]
    if isinstance(node, ast.UnaryOp):
        return [type(node.op).__name__, operator_tree(node.operand)]
    return ast.unparse(node)


def source_facts(source):
    """Parse only. No candidate imports, evaluation, rules for preferred math, or gate."""
    result = {'source_sha256': source_hash(source)}
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return dict(result, parse_error={'message': error.msg, 'line': error.lineno,
                                         'offset': error.offset}, units={})
    units = {}
    module = []

    def add(name, node):
        # All binary operators are represented, not a curated list of graph mechanisms.
        units[name] = {'ast': ast.dump(node, include_attributes=False),
                       'line': getattr(node, 'lineno', None),
                       'operators': [{'line': n.lineno, 'expression': ast.unparse(n),
                                      'grouping': operator_tree(n)}
                                     for n in ast.walk(node) if isinstance(n, ast.BinOp)]}

    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == 'GRAPH_SCORE_CONTRACT' for t in targets):
            continue  # Executor identity metadata, retained in exact source hashes/diffs.
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(node.name + '.' + child.name, child)
            shell = copy.deepcopy(node)
            shell.body = [n for n in shell.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            add(node.name + '.<class>', shell)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, node)
        else:
            module.append(node)
    add('<module>', ast.Module(body=module, type_ignores=[]))
    return dict(result, parse_error=None, units=units)


def compare_sources(reference, candidate):
    before, after = source_facts(reference), source_facts(candidate)
    changed = []
    for name in sorted(before['units'].keys() | after['units'].keys()):
        left, right = before['units'].get(name), after['units'].get(name)
        if (left or {}).get('ast') != (right or {}).get('ast'):
            changed.append({'unit': name, 'change': 'added' if left is None else 'removed' if right is None else 'modified',
                            'reference': {k: v for k, v in (left or {}).items() if k != 'ast'},
                            'draft': {k: v for k, v in (right or {}).items() if k != 'ast'}})
    return {'reference_source_sha256': before['source_sha256'],
            'draft_source_sha256': after['source_sha256'],
            'reference_parse_error': before['parse_error'], 'draft_parse_error': after['parse_error'],
            'changed_units': changed,
            'unified_diff': ''.join(difflib.unified_diff(reference.splitlines(True), candidate.splitlines(True),
                                                       fromfile='exact-reference.py', tofile='unexecuted-draft.py')),
            'limit': 'AST inequality is not a proof of functional inequality. Identity metadata excluded only from AST units.'}


def review_packet(adapter, draft, grounded):
    packet = {'stage': 'review', 'instruction': REVIEW, 'draft': copy.deepcopy(draft),
              'draft_digest': digest(draft), 'measured_draft_result': None}
    if grounded:
        source = draft['program']['source']
        comparison = {'instruction': GROUNDING,
                      'root_id': adapter.parent_id,
                      'relative_to_root': compare_sources(adapter.parent, source)}
        try:
            parent = adapter._parent_source(draft['closest_parent'])
            comparison['declared_parent_hash_matches'] = source_hash(parent) == draft['program']['parent_source_sha256']
            comparison['relative_to_declared_parent'] = (
                {'same_reference_as_root': True} if parent == adapter.parent else compare_sources(parent, source))
        except ValueError as error:
            comparison['declared_parent_unavailable'] = str(error)
        packet['code_comparison'] = comparison
    return packet


def install_two_pass(producer, adapter, grounded):
    """Keep native receipts/accounting for both passes, including unselected drafts."""
    original = producer._call
    provider_call = producer.provider_call

    def sealed_call(**options):
        # Namespaced stage calls must obey the same exact-request recovery as native
        # single-stage calls. Never re-render a started request under its old identity.
        from recclaw_core.research_line.provider import _stored_provider_request_replay, SealedProviderRequestUnavailableError
        if 'resume_request_digest' not in options:
            started, request_digest = _stored_provider_request_replay(
                options['call_root'], logical_call_id=options['logical_call_id'], session_id=options['session_id'])
            if started and request_digest is None:
                raise SealedProviderRequestUnavailableError('Started revision stage lost its sealed request')
            if request_digest is not None:
                options['resume_request_digest'] = request_digest
        return provider_call(**options)

    producer.provider_call = sealed_call

    def configure_replay(traces):
        # Native one-response-per-role assumption no longer applies: each role has
        # two explicit paid stages. Bind every exact logical request independently.
        identities = {}
        for trace in traces:
            call_id = trace.get('logical_call_id', '')
            receipt = trace.get('receipt') or {}
            request_digest = receipt.get('request_digest')
            if request_digest:
                if not any('gr-v1-' + stage in call_id for stage in STAGES):
                    raise ValueError('Cannot import another producer workflow into revision replay')
                if call_id in identities and identities[call_id] != request_digest:
                    raise ValueError('Conflicting sealed request identity')
                identities[call_id] = request_digest
        producer._resume_request_digest_by_logical_call = identities
        producer._resume_request_by_producer_role = {}

    producer.configure_resume_provider_replay = configure_replay

    def call(role, view, *, logical_namespace=None, resume_trace=None):
        view = program_projection(view)
        key = digest({'context': view, 'role': role, 'grounded': grounded})
        path = Path(producer.call_root) / 'revision_stages' / (key + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        state = json.loads(path.read_bytes()) if path.exists() else {'context_digest': key, 'stages': {}}
        retry_stage = None
        if resume_trace is not None:
            identity = resume_trace.get('logical_call_id', '')
            matches = [stage for stage in STAGES if 'gr-v1-' + stage in identity]
            if len(matches) != 1:
                raise ValueError('Transport continuation has no unique revision stage')
            if (resume_trace.get('receipt') or {}).get('status') != 'SUCCESS':
                if not isinstance((resume_trace.get('sealed_request') or {}).get('request_payload'), dict):
                    raise ValueError('Missing exact sealed payload for failed stage continuation')
                retry_stage = matches[0]

        def save():
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            temporary.replace(path)

        def stage_call(stage, stage_view):
            namespace = state['stages'].get(stage, {}).get('namespace', 'gr-v1-' + stage)
            trace = resume_trace if retry_stage == stage else None
            if trace is not None:
                namespace = 'gr-v1-' + stage + ':' + str(logical_namespace)
            elif logical_namespace and resume_trace is None:
                namespace = 'gr-v1-' + stage + ':' + logical_namespace
            state['stages'][stage] = {'namespace': namespace, 'input_digest': digest(stage_view)}
            save()
            result = original(role, stage_view, logical_namespace=namespace, resume_trace=trace)
            state['stages'][stage].update(proposal=result, trace=producer.last_call_trace)
            save()
            return result

        draft_view = copy.deepcopy(view)
        draft_view['knowledge_base']['search_space']['revision_pass'] = {
            'stage': 'draft', 'instruction': 'Produce one complete candidate draft; the same role will have one final review pass before native selection.'}
        draft = stage_call('draft', draft_view)
        revised_view = copy.deepcopy(view)
        packet = review_packet(adapter, draft, grounded)
        revised_view['knowledge_base']['search_space']['revision_pass'] = packet
        state['review_packet'] = packet
        save()
        return stage_call('review', revised_view)

    producer._call = call
    return producer


@contextmanager
def revision_scope(grounded):
    class RevisionAdapter(ProgramAdapter):
        def build_provider_research_producer(self, **kwargs):
            wrapped = super().build_provider_research_producer(**kwargs)
            install_two_pass(wrapped._producer, self, grounded)
            # Native transport continuation otherwise skips GraphProviderContextAdapter.
            wrapped.resume_transport_call = lambda role, view, **kw: wrapped._producer.resume_transport_call(
                role, self.provider_context(view), **kw)
            return wrapped

    import program_adapter
    # program_scope retains deterministic source handoff, no import repair and no
    # second Implementer model. Both arms share this exact path and root semantics.
    with patch.object(program_adapter, 'ProgramAdapter', RevisionAdapter), program_scope():
        yield
