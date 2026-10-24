"""Build offline three-arm inputs from original frozen request facts, without API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import tiktoken

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--requests', type=Path, required=True)
parser.add_argument('--state18-original', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
encoder = tiktoken.get_encoding('o200k_base')
rows = []
for state in (2, 3, 18, 6, 10, 8, 12):
    requests, facts = {}, {}
    for arm in ('Summary', 'Mechanism'):
        p = args.requests / f'state_{state}' / f'{arm}.json'
        q = json.loads(p.read_text(encoding='utf-8-sig'))
        f = json.loads(q['prompt'].split('\nCOMMON_FACTS_JSON\n', 1)[1].split('\nRESPONSE_SCHEMA_JSON\n', 1)[0])
        assert digest(f) == q['common_facts_sha256'], (state, arm)
        requests[arm], facts[arm] = q, f
    assert facts['Summary'] == facts['Mechanism'], state
    for key in ('model', 'reasoning_effort', 'max_output_tokens'):
        assert requests['Summary'][key] == requests['Mechanism'][key], (state, key)
    # Differential batches include their representation RULE in the historical
    # action hash. Compare actual shared permissions before REPRESENTATION.
    permissions = requests['Summary']['prompt'].split('\nREPRESENTATION\n', 1)[0]
    assert permissions == requests['Mechanism']['prompt'].split('\nREPRESENTATION\n', 1)[0], state
    out = args.output_dir / f'state_{state}'
    out.mkdir(parents=True, exist_ok=True)
    common = facts['Summary']
    (out / 'COMMON_FACTS.json').write_text(canonical(common)+'\n', encoding='utf-8')
    raw = {'stage':'downstream common-facts input; no feedback-generation request',
           'common_facts_sha256':digest(common), 'facts':common,'representation':None}
    (out / 'Raw.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    entry = {'source_state':state, 'common_facts_sha256':digest(common), 'arms':{},
             'raw_feedback_calls':0, 'common_facts_bytes':len(canonical(common).encode()),
             'shared_permissions_sha256':hashlib.sha256(permissions.encode()).hexdigest(),
             'historical_action_hashes':{a:q['action_contract_sha256'] for a,q in requests.items()}}
    for arm,q in requests.items():
        # Preserve the original complete prompt and matched permissions; no lossy truncation.
        (out / f'{arm}.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        count = len(encoder.encode(q['prompt']))
        entry['arms'][arm] = {'prompt_tokens_o200k_base':count,
            'max_output_tokens':q['max_output_tokens'],
            'prompt_sha256':hashlib.sha256(q['prompt'].encode()).hexdigest(),
            'estimated_reference_usd_at_output_cap':(count*2+q['max_output_tokens']*12)/1e6,
            'original_request_sha256':hashlib.sha256((args.requests/f'state_{state}'/f'{arm}.json').read_bytes()).hexdigest()}
    rows.append(entry)
contract = json.loads((args.state18_original/'PROSPECTIVE_CONTRACT.json').read_text())
raw_bytes = (args.state18_original/'COMMON_M18_RAW.json').read_bytes()
assert hashlib.sha256(raw_bytes).hexdigest() == contract['common_raw_sha256']
cost = sum(a['estimated_reference_usd_at_output_cap'] for r in rows for a in r['arms'].values())
report = {'status':'OFFLINE_PAYLOADS_BUILT_NOT_LAUNCHED', 'states':rows,
    'state18_identity':{k:contract[k] for k in ('common_checkpoint','checkpoint_sha256','common_raw_sha256','seeds','information_cutoff')},
    'state18_history_source':'Original paid request COMMON_FACTS_JSON prior_history; not an invented FROZEN_STATE or HISTORY_MANIFEST file.',
    'feedback_only_stage':{'calls':14,'full_fits':0,'gpu_seconds':0,'estimated_reference_usd_at_output_cap':cost,
        'proposed_reference_cap_usd':round(cost+0.25,2),'allocated':False,
        'scientific_use':'Complete seven-state feedback-generation repeat and evidence-linked decision analysis; not discovery-quality measurement.'},
    'cost_basis':{'input_usd_per_million':2,'output_usd_per_million':12,
        'source':'Original PREFIX_RESULT API normalized prices; reference cost, not current invoice.',
        'tokenizer':'o200k_base estimate; provider framing/tokenization may differ; billable counts unknown until execution.'},
    'remaining':['Bind fresh request namespace and seeds without replaying completed requests.',
        'Build downstream four-role researcher, selector and implementation payloads after feedback exists; current Raw is a facts envelope, not a complete provider request.',
        'Measure downstream upper-bound costs and context capacity; allocate independently before any paid call.',
        'For algorithm quality, complete all 21 candidate evaluations under the same finite limits.'],
    'new_api_calls':0,'new_fits':0}
(args.output_dir/'PAYLOAD_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'states':7,'matched_feedback_pairs':7,'raw_fact_envelopes':7,
    'prompt_tokens_min':min(a['prompt_tokens_o200k_base'] for r in rows for a in r['arms'].values()),
    'prompt_tokens_max':max(a['prompt_tokens_o200k_base'] for r in rows for a in r['arms'].values()),
    'feedback_reference_usd_at_output_cap':cost,'state18_raw_hash_match':True}))
