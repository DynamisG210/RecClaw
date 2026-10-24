"""CPU-only synthetic qualification; JSON input carries source and source root."""
import hashlib
import json
import random
import sys


def main(payload):
    sys.path.insert(0, payload['recbole_root'])
    import numpy as np
    import torch
    torch.set_num_threads(1)
    cfg = dict(n_layers=2, n_heads=2, hidden_size=64, inner_size=256,
               hidden_dropout_prob=.2, attn_dropout_prob=.2, hidden_act='gelu',
               layer_norm_eps=1e-12, mask_ratio=.2, MASK_ITEM_SEQ='masked_item_seq',
               POS_ITEMS='positive_items', NEG_ITEMS='negative_items', MASK_INDEX='masked_index',
               loss_type='CE', initializer_range=.02, USER_ID_FIELD='user_id',
               ITEM_ID_FIELD='item_id', LIST_SUFFIX='_list', ITEM_LIST_LENGTH_FIELD='item_length',
               NEG_PREFIX='neg_', MAX_ITEM_LIST_LENGTH=50, device='cpu')
    class Dataset:
        def num(self, field):
            return 64
    states, reports = {}, {}
    for name, source in payload['sources'].items():
        # The adapter only converts the config interface. This diagnostic uses
        # a native mapping and tests the unchanged candidate class body.
        source_body = source.split('from recclaw_core.experiments.helix_abc_v1.model_configuration')[0]
        ns = {}
        exec(compile(source_body, name, 'exec'), ns)
        torch.manual_seed(51008)
        random.seed(51008)
        np.random.seed(51008)
        model = ns['FreshCandidateModel'](cfg, Dataset())
        states[name] = {k:v.detach().clone() for k,v in model.state_dict().items()}
        rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        batch = ns['training_transform']([[1,2,3,4,5],[6,7,8,9]],64,
                    {'max_seq_length':50, 'mask_ratio':.2},'cpu',epoch=0,batch_index=0)
        loss = model.calculate_loss(batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        grad = None if not hasattr(model,'query_projection') else float(model.query_projection.weight.grad.abs().sum())
        if grad is not None:
            assert grad > 0
        model.eval()
        batch['item_id'] = torch.tensor([10,11])
        with torch.no_grad():
            scores = model.full_sort_predict(batch)
            predicted = model.predict(batch)
        assert torch.isfinite(scores).all()
        gap = float((predicted-scores[torch.arange(2),batch['item_id']]).abs().max())
        assert gap < 1e-6
        reports[name] = dict(loss=float(loss.detach()), projection_gradient_l1=grad,
                             predict_fullsort_max_gap=gap, post_init_rng=rng,
                             parameters=sum(p.numel() for p in model.parameters()))
    pairs = {}
    for left,right in [('combination','score_only'),('root','objective_only'),('root','combination')]:
        common = sorted(set(states[left]) & set(states[right]))
        mismatches = [k for k in common if not torch.equal(states[left][k],states[right][k])]
        pairs[left+'__'+right] = dict(common_count=len(common), mismatched_parameters=mismatches)
    assert not pairs['combination__score_only']['mismatched_parameters']
    print(json.dumps(dict(kind='SYNTHETIC_CPU_NOT_TRAINING', torch=torch.__version__,
                          reports=reports, initialization_pairs=pairs, new_fit=0, new_test=0,
                          config_adapter_excluded=True), indent=2))


if __name__ == '__main__':
    main(json.load(sys.stdin))
