"""Run-local genuine evaluation capacity and common primary-metric winner.

Trusted evaluator outcomes only. This is one MAIN consumer, not API/GPU/active
wall accounting. Those consumers must supply output admission before MAIN use.
"""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import time
import uuid


class EvaluationLimitReached(RuntimeError):
    pass


class EvaluationLedger:
    def __init__(self, path, *, root_sha256, identity, maximum=12):
        self.path = Path(path)
        self.contract = dict(root_sha256=root_sha256, identity=identity, maximum=maximum)

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                state = json.loads(self.path.read_text())
                if state['contract'] != self.contract:
                    raise ValueError('Evaluation ledger belongs to another repeat/protocol')
            else:
                state = dict(contract=self.contract, pending={}, results=[], genuine_count=0, winner=None,
                             next_available_index=0)
            yield state
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, sort_keys=True, indent=2))
            temporary.replace(self.path)

    def reserve(self, source_sha256):
        with self.transaction() as state:
            if any(row['source_sha256'] == source_sha256 for row in state['pending'].values()):
                raise EvaluationLimitReached('Existing evaluation is unresolved; recover it without another worker')
            if state['genuine_count'] >= self.contract['maximum']:
                raise EvaluationLimitReached('MAIN genuine evaluation limit reached')
            known = {r['source_sha256'] for r in state['results'] if r['genuine']}
            is_new = source_sha256 != self.contract['root_sha256'] and source_sha256 not in known
            reserved = {r['source_sha256'] for r in state['pending'].values() if r['new']}
            if is_new and source_sha256 not in reserved and state['genuine_count'] + len(reserved) >= self.contract['maximum']:
                raise EvaluationLimitReached('MAIN genuine evaluation capacity exhausted')
            token = uuid.uuid4().hex
            state['pending'][token] = dict(source_sha256=source_sha256, new=is_new)
            return token

    def ready(self, token, result, *, admitted):
        """Checkpoint actual returned output before delivering/counting it."""
        with self.transaction() as state:
            row = state['pending'][token]
            if result['artifacts']['source_sha256'] != row['source_sha256']:
                raise ValueError('Ready result does not belong to its reservation')
            if 'result' in row:
                if row['result'] != result or row['admitted'] != admitted:
                    raise ValueError('Ready result replay differs')
                return
            state.setdefault('next_available_index', len(state['results']))
            row.update(result=result, admitted=admitted, ready_ns=time.time_ns(),
                       available_index=state['next_available_index'])
            state['next_available_index'] += 1

    def recoverable(self, source_sha256):
        with self.transaction() as state:
            for token, row in state['pending'].items():
                if row['source_sha256'] == source_sha256 and 'result' in row:
                    return dict(token=token, **row)
            for row in state['results']:
                if row['source_sha256'] == source_sha256 and row['valid'] and 'result' in row:
                    return row
        return None

    def finish(self, token, *, valid, primary_metric, admitted, evidence):
        """Retain late/failure evidence; only on-time legal finite results count.

        `admitted` comes from the trusted common resource supervisor, never the
        candidate. Completion order is availability order for equal-score ties.
        """
        if valid and (not isinstance(primary_metric, (int, float)) or isinstance(primary_metric, bool)
                      or not math.isfinite(primary_metric)):
            raise ValueError('Valid evaluation needs a finite trusted primary metric')
        with self.transaction() as state:
            existing = next((r for r in state['results'] if r['token'] == token), None)
            if existing:
                if (existing['valid'], existing['primary_metric'], existing['admitted'], existing['evidence']) != (valid, primary_metric, admitted, evidence):
                    raise ValueError('Completed evaluation replay differs')
                return existing
            reservation = state['pending'].pop(token)
            if 'result' in reservation and (reservation['admitted'] != admitted
                    or reservation['result']['artifacts']['native_evidence'] != evidence):
                raise ValueError('Delivery differs from the original ready output')
            source = reservation['source_sha256']
            is_root = source == self.contract['root_sha256']
            duplicate = any(r['source_sha256'] == source and r['genuine'] for r in state['results'])
            genuine = bool(valid and admitted and not is_root and not duplicate)
            if genuine and state['genuine_count'] >= self.contract['maximum']:
                raise EvaluationLimitReached('MAIN evaluation completion exceeded reserved capacity')
            state.setdefault('next_available_index', len(state['results']))
            available_index = reservation.get('available_index', state['next_available_index'])
            if 'available_index' not in reservation:
                state['next_available_index'] += 1
            row = dict(token=token, source_sha256=source, valid=valid,
                primary_metric=primary_metric, admitted=admitted, root=is_root,
                duplicate=duplicate, genuine=genuine, evidence=evidence,
                available_index=available_index)
            if 'result' in reservation:
                row.update(result=reservation['result'], ready_ns=reservation['ready_ns'])
            state['results'].append(row)
            state['genuine_count'] += int(genuine)
            current = state['winner']
            # Root wins equal-score ties even when it is loaded after a candidate.
            if valid and admitted and (current is None or primary_metric > current['primary_metric']
                or (primary_metric == current['primary_metric'] and is_root and not current['root'])
                or (primary_metric == current['primary_metric'] and is_root == current['root']
                    and available_index < current['available_index'])):
                state['winner'] = row
            return row

    def snapshot(self):
        with self.transaction() as state:
            return state


class MainResultConsumer:
    """Attach to the actual program evaluator without adding model/training calls."""
    def __init__(self, evaluator, ledger, output_admission, start_admission=None):
        self.evaluator, self.ledger = evaluator, ledger
        self.output_admission = output_admission
        self.start_admission = start_admission
        self.runtime = evaluator.runtime

    def evaluate(self, program_path):
        import hashlib
        source = Path(program_path).read_bytes().decode('utf-8')
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        from resource_stop import SearchResourceStop
        recovered = self.ledger.recoverable(digest)
        if recovered is not None:
            # Settle the ORIGINAL on-time output, not a new evaluation at the
            # resume time. Provider admission still prevents new work after stop.
            return self._deliver(recovered['token'], recovered['result'], recovered['admitted'])
        if self.start_admission is not None:
            self.start_admission()
        try:
            token = self.ledger.reserve(digest)
        except EvaluationLimitReached as stop:
            raise SearchResourceStop(str(stop)) from stop
        # Interrupted work stays pending until its real result is recovered;
        # do not silently free slots or retry completed native work here.
        from gagc.e1_runtime import BudgetExhausted
        try:
            result = self.evaluator.evaluate(program_path)
        except (BudgetExhausted, SearchResourceStop) as stop:
            self.ledger.finish(token, valid=False, primary_metric=None, admitted=False,
                               evidence={'resource_stop': str(stop)})
            if isinstance(stop, SearchResourceStop):
                raise
            raise SearchResourceStop(str(stop)) from stop
        artifacts = result['artifacts']
        if artifacts['source_sha256'] != digest:
            raise ValueError('Program source changed during native evaluation')
        admitted = self.output_admission()
        self.ledger.ready(token, result, admitted=admitted)
        return self._deliver(token, result, admitted)

    def _deliver(self, token, result, admitted):
        from resource_stop import SearchResourceStop
        artifacts = result['artifacts']
        valid = artifacts['valid']
        self.ledger.finish(token, valid=valid,
            primary_metric=result['metrics'].get('ndcg_at_5') if valid else None,
            admitted=admitted, evidence=artifacts['native_evidence'])
        if not admitted:
            raise SearchResourceStop('Result retained as evidence but arrived after common resource deadline')
        return result

    def openevolve_result(self, program_path):
        from openevolve.evaluation_result import EvaluationResult
        result = self.evaluate(program_path)
        return EvaluationResult(metrics=result['metrics'], artifacts={
            'native_evidence.json': json.dumps(result['artifacts'], sort_keys=True)})
