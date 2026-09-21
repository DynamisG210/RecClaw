"""Native result/program ports over the shared graph evaluator.

No research controller is implemented here. SE cannot be constructed without
its real observation/feedback callback. Common winner remains raw NDCG@5.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

START='# EVOLVE-BLOCK-START\n'
END='# EVOLVE-BLOCK-END\n'
MARKER=re.compile(r'^# === ([^\r\n]+) ===\r?$',re.MULTILINE)


def single_source(program,parent):
    if program.startswith(START):
        if not program.endswith(END): raise ValueError('Unclosed native edit block')
        program=program[len(START):-len(END)]
    # Normalize only native transport framing of an unchanged original parent.
    if program.replace('\r\n','\n').strip()==parent.replace('\r\n','\n').strip(): return parent
    return program


def native_files(program,parent,method):
    matches=list(MARKER.finditer(program))
    if not matches: return {'candidate.py':single_source(program,parent['candidate.py'])}
    files={}
    for index,match in enumerate(matches):
        name=match.group(1)
        if name in files: raise ValueError('Duplicate native graph source file')
        end=matches[index+1].start() if index+1<len(matches) else len(program)
        value=program[match.end():end].lstrip('\r\n').rstrip()+'\n'
        if name in parent and value.strip()==parent[name].replace('\r\n','\n').strip(): value=parent[name]
        files[name]=value
    expected={'candidate.py','diagnosis_tools.py'} if method=='se' else {'candidate.py'}
    if set(files)!=expected: raise ValueError('Native graph carrier must retain '+str(sorted(expected)))
    files['candidate.py']=single_source(files['candidate.py'],parent['candidate.py'])
    return files


def evaluate_source(config,source):
    from graph_candidate_evaluator import run_program
    parent=Path(config['parent_source']).read_bytes().decode('utf-8')
    source=single_source(source,parent)
    if source==parent: return run_program(config)
    digest=hashlib.sha256(source.encode()).hexdigest()
    path=Path(config['output'])/'programs'/(digest+'.py');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=source.encode(): raise ValueError('Saved graph candidate bytes differ')
    else: path.write_bytes(source.encode())
    return run_program(config,path)


def evaluate(program_path):
    """Official OpenEvolve evaluator entry: text artifacts, native fitness ABI."""
    config=json.loads(Path(os.environ['E1_RQ1_EVALUATOR_CONFIG']).read_bytes())
    result=evaluate_source(config,Path(program_path).read_bytes().decode('utf-8'))
    from openevolve.evaluation_result import EvaluationResult
    return EvaluationResult(metrics=result['metrics'],artifacts={
        'native_evidence.json':json.dumps(result['artifacts'],sort_keys=True)})


def evaluate_shinka(config,program_path,results_dir):
    from graph_candidate_evaluator import configure
    configure(config,worker_process=False)
    from durable_evaluation import write_json
    from resource_stop import SearchResourceStop
    output=Path(results_dir);output.mkdir(parents=True,exist_ok=True)
    try: result=evaluate_source(config,Path(program_path).read_bytes().decode('utf-8'))
    except SearchResourceStop as stop:
        write_json(output/'RESOURCE_STOP.json',dict(reason=str(stop)))
        raise
    metrics,artifacts=result['metrics'],result['artifacts']
    write_json(output/'native_evidence.json',artifacts)
    write_json(output/'metrics.json',dict(combined_score=metrics['combined_score'],
        public={'ndcg_at_5':metrics['ndcg_at_5']} if artifacts['valid'] else {},private={},
        text_feedback=json.dumps(artifacts['native_evidence'],ensure_ascii=False)))
    write_json(output/'correct.json',dict(correct=artifacts['valid']))


def research_metrics(envelope,method,feedback=None):
    if not envelope['artifacts']['valid']: raise ValueError('No completed native graph evaluation')
    outcome=envelope['artifacts']['native_evidence']
    score=outcome['metrics']['ndcg@5']
    if not math.isfinite(score) or not 0<=score<=1 or outcome['test_accessed'] is not False:
        raise ValueError('Invalid graph validation evidence')
    if method=='deep': return dict(combined_score=score,ndcg_at_5=score)
    if method!='se': raise ValueError('Expected SE or Deep native Problem')
    hit=outcome['metrics']['hit@5']
    if not math.isfinite(hit) or not 0<=hit<=1: raise ValueError('SE requires measured Hit@5')
    if feedback is None or len(feedback)!=3 or type(feedback[0]) is not bool:
        raise ValueError('SE requires actual critical/simulation/diagnosis feedback')
    critical,simulation,diagnosis=feedback
    if not isinstance(simulation,str) or not simulation.strip() or not isinstance(diagnosis,str) or not diagnosis.strip():
        raise ValueError('SE feedback must contain actual native comments')
    return dict(outcome['metrics'],combined_score=0. if critical else .6*hit+.4*score,
        ndcg_at_5=score,hit_at_5=hit,simulator_comment=simulation,diagnosis_comment=diagnosis)


def bound_problem_class(native_problem,config,parent,*,method,feedback_call=None):
    if method=='se' and feedback_call is None: raise ValueError('Bind real SE feedback first')
    class GraphProblem(native_problem):
        async def evaluate(self,code,program_id,is_initial=False):
            current,retries=code,0
            while retries<=self.max_retry_times:
                try: files=native_files(current,parent,method)
                except (ValueError,SyntaxError) as error: success,message=False,str(error)
                else:
                    if method=='se' and set(files)!={'candidate.py','diagnosis_tools.py'}:
                        raise ValueError('SE must retain its separate evolved diagnostic source')
                    if is_initial and files!=parent: raise ValueError('Initial graph program differs from frozen parent')
                    started=time.monotonic()
                    # Controller async lifecycle stays native; physical request
                    # and result are independently durable in the common evaluator.
                    result=await asyncio.to_thread(evaluate_source,config,files['candidate.py'])
                    success=result['artifacts']['valid']
                    message=json.dumps(result['artifacts']['native_evidence'],ensure_ascii=False)
                    if success:
                        comments=None
                        if method=='se':
                            success,comments=await feedback_call(files,result['artifacts']['native_evidence'])
                            if not success: message=str(comments)
                        if success:
                            scores=research_metrics(result,method,comments)
                            self.execution_time.append(time.monotonic()-started)
                            return scores,current
                if is_initial: raise RuntimeError('Initial graph evaluation/feedback failed: '+message)
                success,current,retries=await self._debugging(current,message,retries)
                if not success: break
            failure={'combined_score':0.}
            if method=='se': failure.update(simulator_comment='Evaluation failed: '+message,
                                           diagnosis_comment='No completed native diagnosis: '+message)
            return failure,current
    return GraphProblem


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--program_path',type=Path,required=True)
    parser.add_argument('--results_dir',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_bytes())
    from graph_candidate_evaluator import configure
    configure(config,worker_process=False)
    from resource_stop import SearchResourceStop
    try: evaluate_shinka(config,args.program_path,args.results_dir)
    except SearchResourceStop: return 75
    return 0


if __name__=='__main__': sys.exit(main())
