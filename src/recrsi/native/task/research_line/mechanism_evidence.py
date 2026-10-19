"""Retain own-run frontier evidence; never substitute the construction parent."""
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid


def identity(candidate_id):
    return hashlib.sha256(candidate_id.encode()).hexdigest()


def augment_evidence(context, output):
    result = copy.deepcopy(context)
    archive = Path(output) / 'research_execution_evidence'
    latest = result.get('latest_completed_execution')
    if latest is not None:
        candidate = latest['candidate_id']
        files = latest['implementation']['files']
        source = [f['content'] for f in files if f['path'] == 'candidate.py']
        if len(source) != 1 or hashlib.sha256(source[0].encode()).hexdigest() != latest['entrypoint_source_sha256']:
            raise ValueError('Measured execution source identity mismatch')
        archive.mkdir(parents=True, exist_ok=True)
        path = archive / (identity(candidate) + '.json')
        if path.exists():
            prior = json.loads(path.read_bytes())
            if (prior['candidate_id'], prior['entrypoint_source_sha256']) != (candidate, latest['entrypoint_source_sha256']):
                raise ValueError('Execution evidence identity changed')
        else:
            temporary = archive / (uuid.uuid4().hex + '.tmp')
            temporary.write_text(json.dumps(latest, ensure_ascii=False), encoding='utf8')
            os.replace(temporary, path)
    frontier = result.get('frontier') or {}
    incumbent = dict(candidate=frontier.get('incumbent_candidate_id'),
                     round=frontier.get('incumbent_round_index'),
                     value=frontier.get('incumbent_ndcg@5'))
    candidate = incumbent.get('candidate')
    entry = dict(status='NO_ARCHIVED_INCUMBENT', incumbent=incumbent)
    if candidate:
        path = archive / (identity(candidate) + '.json')
        if path.exists():
            execution = json.loads(path.read_bytes())
            if execution['candidate_id'] != candidate:
                raise ValueError('Incumbent archive identity mismatch')
            entry = dict(status='AVAILABLE', incumbent=incumbent, execution=execution,
                         semantics='Own-run observation, not a new evaluation or implicit construction parent.')
    result['knowledge_base']['search_space']['incumbent_execution_evidence'] = entry
    return result
