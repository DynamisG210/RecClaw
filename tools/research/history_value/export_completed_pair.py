"""Read trusted native checkpoints and ledgers; never call a provider or worker."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def export(home, original):
    sys.path[:0] = [str(home / 'runtime'), str(original),
                   *read(original / 'ORIGINAL_RUNTIME_PATHS.json')]
    output = {'schema': 'recrsi.completed-history-pair.v1', 'arms': {}}
    marker = 'Use this complete role-scoped Research Context JSON as the authoritative input for this call:'
    for arm in ('retain', 'reset'):
        root = home / arm / 'search'
        stage_path = root / 'STAGE_RESULT.json'
        stage = read(stage_path)
        events = [json.loads(line) for line in (root / 'API_EVENTS.jsonl').read_text().splitlines()]
        traces = []
        boundaries = []
        for round_index in (13, 14, 15):
            path = root / 'campaign' / f'ROUND_{round_index:02d}_PREPARED_CHECKPOINT.pkl'
            prepared = pickle.loads(path.read_bytes())['prepared']
            current = list(prepared.provider_traces)
            # The first paid producer batch was reused after construction binding
            # repair; its traces live in the preserved recovery artifact.
            if arm == 'retain' and round_index == 13:
                paid = pickle.loads((root / 'FIRST_PAID_BATCH_RECOVERY.pkl').read_bytes())
                current += list(paid.provider_traces)
            traces.extend(current)
            for trace in current:
                if trace['kind'] != 'research_producer':
                    continue
                payload = trace['sealed_request']['request_payload']
                prompt = payload['input']
                context, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())
                boundaries.append({'round': round_index,
                    'role': trace['receipt']['provider_role'],
                    'request_digest': trace['sealed_request']['request_digest'],
                    'input_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'request_payload': payload,
                    'context': context})
        unique = {t['logical_call_id']: t for t in traces}
        successes = [e for e in events if e['event'] == 'SUCCESS']
        matched = set()
        matches = []
        for trace in unique.values():
            if trace['receipt']['status'] == 'SUCCESS':
                usage = trace['usage']
                model = trace['receipt']['returned_model']
            else:
                usage = trace.get('failure') or {}
                if usage.get('usage_charge_basis') != 'ACTUAL':
                    continue
                model = trace['receipt']['requested_model']
            candidates = [e for e in successes if
                e['input_tokens'] == usage['input_tokens'] and
                e['output_tokens'] == usage['output_tokens'] and e['model'] == model]
            if len(candidates) != 1:
                raise ValueError(f'Nonunique usage mapping: {arm} {trace["logical_call_id"]} '
                                 f'{model} {usage} matches={len(candidates)}')
            event = candidates[0]
            if event['call_id'] in matched:
                raise ValueError('API event assigned to two logical calls')
            matched.add(event['call_id'])
            matches.append({'logical_call_id': trace['logical_call_id'],
                'api_call_id': event['call_id'], 'kind': trace['kind'],
                'native_status': trace['receipt']['status'],
                'failure_class': (trace.get('failure') or {}).get('failure_class'),
                'request_digest': trace['sealed_request']['request_digest'],
                'mapping_basis': 'unique model/input/output usage tuple'})
        output['arms'][arm] = {'stage': stage,
            'stage_sha256': hashlib.sha256(stage_path.read_bytes()).hexdigest(),
            'api_events': events, 'used_call_matches': matches,
            'unmatched_success_ids': [e['call_id'] for e in successes if e['call_id'] not in matched],
            'producer_boundaries': boundaries,
            'frontier': read(root / 'campaign/CAMPAIGN_STATE.json')['state']['frontier']}
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('home', type=Path)
    parser.add_argument('original', type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.home, args.original), ensure_ascii=False))
