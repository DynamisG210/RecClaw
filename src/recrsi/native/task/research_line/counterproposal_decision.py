"""Prospective graph research policy: compare two programs, execute one.

Install before fresh composition. No provider, campaign or fit is launched here.
The native paid response retains both alternatives; only the selected program is
projected into the existing source/qualification/training path.
"""
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

VERSION = 'graph-counterproposal-decision-v4-declared-count'
INSTRUCTION = '''
COMPETING COMPLETE COMPUTATIONS FOR THIS ONE RESEARCH OPPORTUNITY:
Return EXACTLY two complete proposal objects in the proposals array: the first
is PROPOSAL, the second is COUNTERPROPOSAL. Put research_decision beside proposals
at the TOP LEVEL of the response, not inside either proposal or inside the array.
research_decision contains selected, comparison and evidence_updates. Each program
is independently complete; neither proposal object contains another proposal.
Both alternatives use the identical task, role, parent
registry, worker contract and visible information. It must test a materially
different explanation or coupled computation, not merely rename the first code.
Neither complexity, simplicity, distance, novelty nor any named operator receives
a preference. Do not introduce a known competitor or hidden historical program.
Compare the two actual programs before choosing. In research_decision, select
PROPOSAL or COUNTERPROPOSAL and explain why that whole computation offers better
expected end-budget frontier quality given its likely execution cost and the
remaining budget. Consider propagation, representation scale, aggregation,
scoring and loss together wherever they interact; no particular recipe is given.
Use evidence_updates to identify supplied own-run observations and how each
changes the relative case for the two programs. Keep it empty when no relevant
observation is supplied; do not invent results or infer hidden source. A mixed
change measures a combination, not every component separately. Explain negative
evidence and unresolved alternatives without banning an entire mechanism family.
Both alternatives are unexecuted proposals. Only your selected complete proposal
will enter the original candidate path. The other receives no fit, metric,
success status or future execution reservation. Do not merge the two programs.
One original model call and one ordinary opportunity are used. The original
token, API and physical budgets remain binding; keep both programs within them.
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def extend_schema(schema):
    schema = copy.deepcopy(schema)
    item = schema['properties']['proposals']['items']
    text = {'type': 'string', 'minLength': 1}
    update = {'type': 'object', 'additionalProperties': False, 'properties': {
        # Evidence may cite another visible own-run candidate even when the
        # construction route restricts parent_ref to one current branch.
        'candidate_ref': text,
        'observation': text,
        'implication_for_proposal': text,
        'implication_for_counterproposal': text}}
    update['required'] = list(update['properties'])
    decision = {'type': 'object', 'additionalProperties': False, 'properties': {
        'selected': {'enum': ['PROPOSAL', 'COUNTERPROPOSAL']},
        'comparison': text,
        'evidence_updates': {'type': 'array', 'items': update}}}
    decision['required'] = list(decision['properties'])
    schema['properties']['proposals'].update(minItems=2, maxItems=2)
    schema['properties']['research_decision'] = decision
    schema['required'].append('research_decision')
    return schema


def consume_choice(response, *, expected_role):
    """Consume a paid decision without rewriting either program or its claims."""
    first, alternative = copy.deepcopy(response['proposals'])
    decision = copy.deepcopy(response['research_decision'])
    if first['producer_role'] != expected_role or alternative['producer_role'] != expected_role:
        raise ValueError('Counterproposal changed the assigned research role')
    selected = decision['selected']
    if selected not in ('PROPOSAL', 'COUNTERPROPOSAL'):
        raise ValueError('Unknown research decision')
    chosen = first if selected == 'PROPOSAL' else alternative
    record = {'version': VERSION, 'selected': selected, 'decision': decision,
              'proposal_digest': digest(first), 'counterproposal_digest': digest(alternative),
              'selected_proposal_digest': digest(chosen),
              'program_source_sha256': {
                  label: hashlib.sha256(p['program']['source'].encode('utf-8')).hexdigest()
                  for label, p in [('PROPOSAL', first), ('COUNTERPROPOSAL', alternative)]},
              'unselected_status': 'UNEXECUTED_NO_METRIC_NO_RESERVED_OPPORTUNITY'}
    record['identical_program_bytes'] = (first['program']['source'] == alternative['program']['source'])
    # Duplicate alternatives are retained as negative decision-diversity evidence;
    # there is no free retry or compiler substitute for the chosen hypothesis.
    return chosen, record


def adapter_class(base):
    class CounterproposalAdapter(base):
        def _proposal_schema_for(self, parents):
            return extend_schema(super()._proposal_schema_for(parents))

        def build_provider_research_producer(self, **kwargs):
            wrapped = super().build_provider_research_producer(**kwargs)
            producer = wrapped._producer
            native_call, native_provider = producer._call, producer.provider_call

            def provider_call(**options):
                # Append after the normal view projection, preserving the effective
                # Full information boundary and any independent DISCOVER projection.
                # The original broker accepts an explicit expected count and checks
                # it against the response independently of JSON Schema. Declare
                # two authored alternatives; physical selection still admits one.
                return native_provider(**{**options, 'prompt': options['prompt'] + '\n' + INSTRUCTION,
                                          'expected_proposal_count': 2})

            def call(role, view, **options):
                from recclaw_core.research_line import provider as native_provider_module
                view = copy.deepcopy(view)
                view['knowledge_base']['search_space']['research_decision_policy'] = VERSION
                root = Path(producer.call_root) / VERSION
                records = []

                def select_one(response, *, kind):
                    # Original _call has already validated the full new schema
                    # and native per-proposal uniqueness before reaching here.
                    if kind != 'research proposal':
                        raise native_provider_module.fresh_r1.FreshR1Error(
                            'Counterproposal selection requires the graph research lane')
                    selected, record = consume_choice(response, expected_role=role)
                    records.append(record)
                    return selected

                with patch.object(producer, 'call_root', root), patch.object(
                        native_provider_module, '_one_proposal', select_one):
                    selected = native_call(role, view, **options)
                # Paid raw response retains BOTH full proposals; original native
                # role validation and normalization run on the chosen proposal.
                if len(records) != 1:
                    raise ValueError('Expected one validated research decision')
                record = records[0]
                trace = producer.last_call_trace or {}
                receipt = trace.get('receipt') or {}
                identity = {'logical_call_id': trace.get('logical_call_id'),
                            'call_id': receipt.get('call_id'),
                            'request_digest': receipt.get('request_digest')}
                record.update(context_ref=view['context_ref'],
                              context_digest=view['context_digest'], role=role,
                              provider_identity=identity)
                key = digest({'context': view['context_digest'], 'role': role,
                              'provider_identity': identity})
                path = root / 'research_decisions' / (key + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if json.loads(path.read_text(encoding='utf-8')) != record:
                        raise ValueError('Saved research decision changed during replay')
                else:
                    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                return selected

            producer.provider_call = provider_call
            producer._call = call
            return wrapped

        def resolve_confirmation(self, kind, primary_binding, context):
            # Only the consumed original-shape proposal crosses native projection.
            # Its source, parent, score declaration and all scientific fields keep
            # the original validators and source handoff; alternatives stay raw.
            original_schema = lambda parents: base._proposal_schema_for(self, parents)
            with patch.object(self, '_proposal_schema_for', original_schema):
                return super().resolve_confirmation(kind, primary_binding, context)

    return CounterproposalAdapter


@contextmanager
def counterproposal_scope():
    """Compose with the existing Full scope, not a replacement campaign."""
    import program_adapter
    with patch.object(program_adapter, 'ProgramAdapter', adapter_class(program_adapter.ProgramAdapter)):
        yield
