"""Apply the actual root token order to a TRAIN-only RecBole dataset.

Cold items can first occur in dev during root dataset construction. Rebuilding
from train alone must apply the saved public ID mapping, never read dev labels
to rediscover that order. Call before data_preparation converts DataFrames.
"""
import hashlib
import json
import numpy as np


def apply_root_mapping(dataset,mapping,expected_sha256):
    observed=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
    if observed!=expected_sha256: raise ValueError('Frozen catalog mapping identity differs')
    for field in ('user_id','item_id'):
        previous=list(map(str,dataset.field2id_token[field]));target=mapping[field]
        if (len(previous)!=len(target) or set(previous)!=set(target)
                or previous[0]!=target[0] or len(set(target))!=len(target)):
            raise ValueError('Train/static catalog is not the original complete root catalog: '+field)
        token_to_id={token:index for index,token in enumerate(target)}
        remap=np.asarray([token_to_id[token] for token in previous],dtype=np.int64)
        for feature_name in ('inter_feat','user_feat','item_feat'):
            frame=getattr(dataset,feature_name,None)
            if frame is not None and field in frame:
                frame[field]=remap[frame[field].to_numpy(dtype=np.int64)]
        if field == 'item_id' and 'item_id_list' in dataset.inter_feat:
            dataset.inter_feat['item_id_list'] = dataset.inter_feat['item_id_list'].map(
                lambda values: remap[np.asarray(values, dtype=np.int64)])
            dataset.field2id_token['item_id_list'] = np.asarray(target, dtype=object)
            dataset.field2token_id['item_id_list'] = token_to_id
        dataset.field2id_token[field]=np.asarray(target,dtype=object)
        dataset.field2token_id[field]=token_to_id
    # RecBole lookup of side rows is positional by mapped ID, including padding.
    for feature_name,field in [('user_feat','user_id'),('item_feat','item_id')]:
        frame=getattr(dataset,feature_name,None)
        if frame is not None:
            frame.sort_values(field,inplace=True);frame.reset_index(drop=True,inplace=True)
            if not np.array_equal(frame[field].to_numpy(),np.arange(len(mapping[field]))):
                raise ValueError('Mapped side-feature rows are not aligned: '+field)
    return observed
