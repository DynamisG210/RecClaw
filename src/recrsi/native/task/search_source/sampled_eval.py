"""Fixed 1-positive/99-negative ranking, shared by DEV and trusted TEST."""
import hashlib
from pathlib import Path
import numpy as np


def load_pool(spec):
    path = Path(spec['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
    pool = np.load(path, allow_pickle=False)
    users, candidates = pool['users'], pool['candidates']
    assert candidates.shape == (len(users), 100)
    assert len(np.unique(users)) == len(users) and np.all(users > 0)
    assert np.all(candidates > 0)
    assert np.all(np.diff(np.sort(candidates, axis=1), axis=1) > 0)
    return users, candidates


def metrics_for(scores, candidates, positives):
    scores = np.asarray(scores)
    assert scores.shape == candidates.shape and np.isfinite(scores).all()
    positives = np.asarray(positives)
    match = candidates == positives[:, None]
    assert np.all(match.sum(axis=1) == 1)
    # Deterministic item-ID tie break, independent of positive placement.
    order = np.lexsort((candidates, -scores), axis=1)
    ranks = np.argmax(np.take_along_axis(match, order, axis=1), axis=1) + 1
    metrics = {}
    for k in (5, 10):
        hit = ranks <= k
        metrics.update({f'hit@{k}': float(hit.mean()), f'recall@{k}': float(hit.mean()),
                        f'ndcg@{k}': float(np.mean(hit / np.log2(ranks + 1))),
                        f'mrr@{k}': float(np.mean(hit / ranks)),
                        f'precision@{k}': float(hit.mean() / k)})
    return metrics


def score_pool(model, config, users, candidates, histories=None):
    import torch
    from recbole.data.interaction import Interaction
    model.eval()
    from graph_worker import clear_cache
    clear_cache(model)
    rows = []
    with torch.no_grad():
        for start in range(0, len(users), 64):
            end = min(start + 64, len(users))
            batch = Interaction({model.USER_ID: torch.as_tensor(users[start:end], dtype=torch.long,
                                                               device=config['device'])})
            if histories is not None:
                batch.update(histories[start:end].to(config['device']))
            all_scores = model.full_sort_predict(batch).view(end-start, model.n_items)
            item_ids = torch.as_tensor(candidates[start:end], dtype=torch.long, device=config['device'])
            rows.append(torch.gather(all_scores, 1, item_ids).cpu().numpy())
    clear_cache(model)
    return np.concatenate(rows)


def evaluate_dev(model, config, valid_data, pool_spec):
    users, candidates = load_pool(pool_spec)
    inter = valid_data._dataset.inter_feat
    labels = dict(zip(inter[model.USER_ID].tolist(), inter[model.ITEM_ID].tolist()))
    assert len(labels) == len(inter) == len(users) and set(labels) == set(map(int, users))
    histories = None
    if hasattr(model, 'ITEM_SEQ'):
        # Reorder complete prefix interactions to the frozen pool's user order.
        index = {int(u): i for i, u in enumerate(inter[model.USER_ID].tolist())}
        from recbole.data.interaction import Interaction
        order = [index[int(u)] for u in users]
        histories = Interaction({key: inter[key][order] for key in (model.ITEM_SEQ, model.ITEM_SEQ_LEN)})
    return metrics_for(score_pool(model, config, users, candidates, histories), candidates,
                       [labels[int(u)] for u in users])
