"""Inventory all seven RQ4 frozen inputs without importing or running experiments."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
states = [
    (2, 'rq4_action_space_two_state_validation_v1_20261009/state_02'),
    (3, 'rq4_action_space_two_state_validation_v1_20261009/state_03'),
    (18, 'rq4_feedback_action_space_v1_20261008'),
    (6, 'rq4_additional_validation_20261014/state_06'),
    (10, 'rq4_additional_validation_20261014/state_10'),
    (8, 'rq4_differential_feedback_v1_20261014/state_08'),
    (12, 'rq4_differential_feedback_v1_20261014/state_12')]
rows = []
for state, directory in states:
    home = args.native_root / directory
    names = ['COMMON_M18_RAW.json' if state == 18 else 'COMMON_RAW.json',
             'FROZEN_STATE.json', 'HISTORY_MANIFEST.json', 'PROSPECTIVE_CONTRACT.json',
             'feedback_contract.py', 'entry.py']
    files = []
    for name in names:
        p = home / name
        item = {'path': str(p), 'exists': p.is_file()}
        if p.is_file():
            raw = p.read_bytes()
            item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        files.append(item)
    frozen = home / 'FROZEN_STATE.json'
    identity = json.loads(frozen.read_text(encoding='utf-8-sig')) if frozen.is_file() else {}
    common = files[0]
    raw_match = common.get('sha256') == identity.get('raw_sha256') if identity and common['exists'] else None
    if raw_match is False:
        raise ValueError(f'State {state}: common input does not match frozen raw hash')
    rows.append({'source_state':state, 'files':files,'raw_matches_frozen':raw_match,
                 'frozen_identity':{k:identity[k] for k in ('round','next_round','checkpoint_sha256',
                    'state_after_digest','actual_parent_id','history_sha256','input_space_sha256') if k in identity}})
result = {'status':'OFFLINE_INPUT_INVENTORY_NOT_EXECUTION_READY', 'states':rows,
          'new_api_calls':0,'new_fits':0,
          'remaining':['Resolve missing local files from original pinned sources.',
                       'Construct and compare complete common-history payloads; do not infer equality from names.',
                       'Bind prospective seeds, evaluator, payload capacity, finite allocation and shared admission before paid execution.']}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'states':len(rows),'raw_hash_matches':sum(r['raw_matches_frozen'] is True for r in rows),
                  'missing_files':[f['path'] for r in rows for f in r['files'] if not f['exists']]}))
