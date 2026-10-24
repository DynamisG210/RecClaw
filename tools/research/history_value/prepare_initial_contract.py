"""Create a prospective input contract and check both recorded serialized inputs offline."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from initial_input_contract import bind_initial_prompt, non_history_view, section, CONTEXT_MARKER


def prepare(data):
    initial = {arm: {b['role']: b['request_payload'] for b in raw['producer_boundaries']
                     if b['round'] == 13} for arm, raw in data['arms'].items()}
    contract = {'schema': 'recrsi.history-initial-input.v2', 'initial_round': 13,
                'roles': {role: non_history_view(payload['input'])
                          for role, payload in initial['reset'].items()}}
    receipt = {'stage': 'OFFLINE_PROSPECTIVE_TRANSFORMATION_ONLY',
               'new_paid_calls': 0, 'new_fits': 0, 'roles': {}}
    for role in contract['roles']:
        role_check = {}
        envelopes = [{k: v for k, v in initial[arm][role].items() if k != 'input'}
                     for arm in ('retain', 'reset')]
        if envelopes[0] != envelopes[1]:
            raise ValueError('Model/output/schema request envelopes differ')
        templates = []
        for arm in ('retain', 'reset'):
            payload = initial[arm][role]
            before = payload['input']
            after = bind_initial_prompt(before, contract)
            original_context, start, end = section(before, CONTEXT_MARKER)
            new_context, new_start, new_end = section(after, CONTEXT_MARKER)
            expected = copy.deepcopy(original_context)
            expected['state']['construction_parent_options'] = contract['roles'][role]['context']['state']['construction_parent_options']
            assert new_context == expected
            assert before[:start] == after[:new_start] and before[end:] == after[new_end:]
            assert non_history_view(after) == contract['roles'][role]
            template = after[:new_start] + '<ROLE_CONTEXT>' + after[new_end:]
            template = re.sub(r'^- (side identity|logical slot):.*$',
                              r'- \1: <ARM_CONTEXT_ID>', template, flags=re.MULTILINE)
            # This native paragraph is generated from the latest historical
            # negative observation. Preserve it in the actual request, but
            # identify it explicitly as a history-mediated instruction.
            recovery = re.findall(r'^EFFECT-FIRST RECOVERY \(weak measured result\):.*\n\n',
                                  template, flags=re.MULTILINE)
            if len(recovery) > 1:
                raise ValueError('Unexpected repeated recovery instruction')
            template = re.sub(r'^EFFECT-FIRST RECOVERY \(weak measured result\):.*\n\n',
                              '', template, flags=re.MULTILINE)
            templates.append(template)
            role_check[arm] = {'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                              'after_sha256': hashlib.sha256(after.encode()).hexdigest(),
                              'history_conditioned_recovery_instruction': recovery,
                              'only_parent_options_changed': True}
        if templates[0] != templates[1]:
            raise ValueError('Prompt text outside role context differs beyond arm identity')
        role_check['common_request_envelope_and_template_equal'] = True
        receipt['roles'][role] = role_check
    receipt['limit'] = ('Checks real recorded serialized inputs through the new binding function; '
                        'not a new native generation or paid result. New homes and separate allocation required.')
    return contract, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    contract, receipt = prepare(json.loads(args.export.read_text(encoding='utf-8-sig')))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('INITIAL_INPUT_CONTRACT_V2.json', contract), ('INPUT_REPAIR_CHECK.json', receipt)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt))
