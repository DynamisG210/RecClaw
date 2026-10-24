"""Prospective first-round parent-input contract; never rewrite paid evidence."""
import copy
import hashlib
import json

CONTEXT_MARKER = 'Use this complete role-scoped Research Context JSON as the authoritative input for this call:'
SOURCE_MARKER = 'Identical initial incumbent source:'


def section(prompt, marker):
    start = prompt.index(marker) + len(marker)
    start += len(prompt[start:]) - len(prompt[start:].lstrip())
    value, size = json.JSONDecoder().raw_decode(prompt[start:])
    return value, start, start + size


def non_history_view(prompt):
    context, _, _ = section(prompt, CONTEXT_MARKER)
    sources, _, _ = section(prompt, SOURCE_MARKER)
    view = copy.deepcopy(context)
    # These are treatment history and identity of the resulting context, not
    # task, source, search ability or parent availability.
    view.pop('memory')
    view['context'].pop('digest')
    view['state'].pop('latest_result', None)
    view['state'].pop('latest_completed_execution', None)
    # The architect observes the other roles' proposals from this same round;
    # this is a downstream response to history, not a frozen baseline input.
    view['state'].pop('research_portfolio', None)
    return {'context': view, 'source_sha256': {
        name: hashlib.sha256(content.encode()).hexdigest() for name, content in sources.items()}}


def bind_initial_prompt(prompt, contract):
    context, start, end = section(prompt, CONTEXT_MARKER)
    if context['context']['round'] != contract['initial_round']:
        return prompt
    role = context['context']['role']
    expected = contract['roles'][role]
    context['state']['construction_parent_options'] = copy.deepcopy(
        expected['context']['state']['construction_parent_options'])
    updated = prompt[:start] + json.dumps(context, ensure_ascii=False, separators=(',', ':')) + prompt[end:]
    if non_history_view(updated) != expected:
        raise ValueError('Non-history initial producer input differs from frozen pair contract')
    return updated


def check_fresh_recovery(root, contract):
    if contract['schema'] != 'recrsi.history-initial-input.v2':
        raise ValueError('Unknown prospective input contract')
    if (root / 'FIRST_PAID_BATCH_RECOVERY.pkl').exists():
        raise ValueError('A pre-contract paid batch cannot seed a new paired stage')
