"""Prepare shared facts only after a measured prospective @5 root exists."""
import copy
from graph_metric_contract import require_binding
from graph_program import common_task_text, primary_score


def shared_facts(original_facts, binding, root_receipt):
    require_binding(binding)
    if root_receipt['status']!='SUCCESS': raise ValueError('Completed new root required')
    result=root_receipt['result']; primary_score(result,binding)
    if (result['train_seed']!=binding['train_seed'] or result['source_sha256']!=binding['root_source_sha256']
            or result['catalog_mapping_sha256']!=binding['catalog_mapping_sha256']):
        raise ValueError('New root must preserve paired seed/source/mapping')
    facts=copy.deepcopy(original_facts)
    facts.update(task=common_task_text(binding),metric_binding_version=binding['metric_binding_version'],
        evaluation=copy.deepcopy(binding['evaluation']),training=copy.deepcopy(binding['training']),
        selected_root_metrics=copy.deepcopy(result['metrics']),
        root_reference_cost=copy.deepcopy(root_receipt['artifacts']['native_evidence']['physical_cost']))
    return facts
