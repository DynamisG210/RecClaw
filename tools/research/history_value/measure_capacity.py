"""Measure reversible JSON interning; no provider request or truncation.

Token counts describe named local encodings, not a provider billing assertion.
"""
import argparse
import json
from pathlib import Path

import tiktoken

from prepare_pair import encoded, digest


def intern(value):
    records, seen = [], {}

    def visit(node):
        if not isinstance(node, (dict, list)):
            return ['scalar', node]
        key = encoded(node)
        if key in seen:
            return ['ref', seen[key]]
        children = ([[k, visit(v)] for k, v in node.items()] if isinstance(node, dict)
                    else [visit(v) for v in node])
        index = len(records)
        records.append(['object' if isinstance(node, dict) else 'array', children])
        seen[key] = index
        return ['ref', index]

    root = visit(value)
    return {'schema': 'recrsi.lossless-json-table.v1', 'root': root, 'records': records}


def expand(table):
    def visit(value):
        tag, content = value
        if tag == 'scalar':
            return content
        kind, children = table['records'][content]
        return ({k: visit(v) for k, v in children} if kind == 'object'
                else [visit(v) for v in children])
    return visit(table['root'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_dir', type=Path)
    args = parser.parse_args()
    report = {'stage': 'LOCAL_CAPACITY_MEASUREMENT', 'actual_provider_request': False,
              'provider_capacity_verified': False, 'new_paid_calls': 0, 'arms': {}}
    encodings = {name: tiktoken.get_encoding(name)
                 for name in ('cl100k_base', 'o200k_base')}
    for arm in ('retain', 'reset'):
        original = json.loads((args.pair_dir / f'{arm}.json').read_bytes())
        packed = intern(original)
        if expand(packed) != original:
            raise ValueError('Lossless round trip failed')
        (args.pair_dir / f'{arm}.interned.json').write_bytes(encoded(packed))
        report['arms'][arm] = {'round_trip_exact': True,
            'original_sha256': digest(original),
            'representations': {label: {'bytes': len(encoded(value)),
                'tokens': {name: len(enc.encode(encoded(value).decode('utf-8'),
                                               disallowed_special=()))
                           for name, enc in encodings.items()}}
                for label, value in (('original', original), ('interned', packed))}}
    (args.pair_dir / 'CAPACITY.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
