"""Verify BERT/ML1M retained checkpoint TEST chains from local native artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--material', type=Path, required=True)
    p.add_argument('--observed', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    observed = read(a.observed)
    summary = read(a.summary)
    rows = []
    for obs in observed['rows']:
        testdir = a.material / 'final_tests' / Path(obs['TEST']['path']).parents[1].name
        tp, cp, fp = testdir/'worker_output/TEST_RESULT.json', testdir/'CONFIG.json', testdir/'WINNER_FREEZE.json'
        test, config, freeze = read(tp), read(cp), read(fp)
        # The archive records paths relative to its original experiment root.
        original_root = config['winner_freeze'].split('/final_tests/',1)[0]+'/'
        def local(remote):
            return a.material / remote[len(original_root):] if remote.startswith(original_root) else None
        paths = {'test':tp,'config':cp,'freeze':fp,'candidate':local(config['candidate']),
                 'checkpoint':local(config['checkpoint'])}
        checks = {
            'test_hash_matches_adopted_observation':digest(tp)==obs['TEST']['sha256'],
            'config_hash_matches_test':digest(cp)==test['config_sha256'],
            'freeze_hash_matches_test':digest(fp)==test['winner_freeze_sha256'],
            'candidate_bytes_match_test':digest(paths['candidate'])==test['candidate_sha256'],
            'candidate_bytes_match_selection':digest(paths['candidate'])==freeze['selection']['source_sha256'],
            'checkpoint_bytes_match_test':digest(paths['checkpoint'])==test['checkpoint_sha256'],
            'checkpoint_matches_frozen_hash':digest(paths['checkpoint'])==freeze['hashes'][config['checkpoint']],
            'adopted_source_matches_original_freeze':obs['selection']['source_sha256']==freeze['selection']['source_sha256'],
            'test_seed_matches_config_and_observation':test['seed']==config['seed']==obs['TEST']['seed'],
            'test_metrics_match_adopted_observation':test['metrics']==obs['TEST']['metrics'],
            'checkpoint_only_no_training':test['training'] is False and config['protocol']['training'] is False,
            'test_feedback_disabled':test['heldout_feedback'] is False and config['protocol']['heldout_feedback'] is False,
        }
        ancillary=[]
        for remote, expected in config['hashes'].items():
            target=local(remote)
            exists=target is not None and target.is_file()
            ancillary.append({'remote_path':remote,'expected_sha256':expected,
                              'status':('MATCH' if digest(target)==expected else 'MISMATCH') if exists else 'NOT_IN_LOCAL_MATERIAL'})
        sel=freeze['selection']
        rows.append({'search_id':obs['unit'],'status':'LOCAL_CORE_CHAIN_VERIFIED' if all(checks.values()) else 'MISMATCH',
                     'checks':checks,'artifacts':{k:{'path':str(v),'sha256':digest(v)} for k,v in paths.items()},
                     'ancillary_hash_checks':ancillary,'selection_rule':sel['rule'],
                     'selected_source':sel['source_sha256'],'native_ndcg_source':sel['native_raw_ndcg_winner_unchanged'],
                     'native_ndcg_selection_equal':sel['source_sha256']==sel['native_raw_ndcg_winner_unchanged'],
                     'original_freeze_genuine':freeze['genuine'],'later_observed_genuine':obs['genuine'],
                     'test_protocol':config['protocol'],'historical_test_exposure':obs['selection']['historical_TEST_exposure'],
                     'limitations':['Evaluation kernel/pool outside material are recorded hashes, not byte-verified here',
                                    'No causal/blind-confirmation inference from post-exposure evidence',
                                    'Later continuation ledger is distinct from original TEST freeze']})
    groups={r['group'] for r in summary['statistics']}
    coverage=[]
    for model in ['SASRec','MoRec','BERT','NGCF']:
        for dataset in ['CDs','Electronics','Office','ML1M']:
            group=model+'_'+dataset
            coverage.append({'group':group,'summary_recomputed':group in groups,
                             'native_core_chains_verified':sum(r['status']=='LOCAL_CORE_CHAIN_VERIFIED' for r in rows) if group=='BERT_ML1M' else 0,
                             'native_scope':'RecRSI RO01-03 only' if group=='BERT_ML1M' else 'NOT_YET_CHECKED',
                             'missing':[] if group in groups else ['source-search inventory','selected candidate identity','checkpoint identity','TEST result','per-search variation']})
    result={'inputs':[{'path':str(x),'sha256':digest(x)} for x in [a.observed,a.summary]],'bert_ml1m':rows,
            'coverage':coverage,'missing_summary_tasks':[r['group'] for r in coverage if not r['summary_recomputed']],
            'new_training':0,'new_test_evaluations':0,'new_api_calls':0}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'chains':[(r['search_id'],r['status'],sum(r['checks'].values()),len(r['checks'])) for r in rows],
                      'ancillary':{s:sum(x['status']==s for r in rows for x in r['ancillary_hash_checks']) for s in ['MATCH','MISMATCH','NOT_IN_LOCAL_MATERIAL']},
                      'missing_summary_tasks':result['missing_summary_tasks']}))

if __name__=='__main__':
    main()
