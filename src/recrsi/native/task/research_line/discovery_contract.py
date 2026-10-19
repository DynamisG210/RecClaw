"""Private Research-only presentation; existing definition/binding consumers stay native."""
import copy
from contextlib import contextmanager,ExitStack
from unittest.mock import patch

VERSION='ngcf-mechanism-discovery-contract-v1'
OLD_DEFINITION_NOTICE="\n\nUse implementation_requirements as the single operational definition of this candidate's changed mathematics and active parameters: specify the relevant operations, dimensions, coefficients, reductions and gradient destinations, including what is replaced and what is inherited. Use the other existing fields for motivation, predictions and alternative tests; refer to that definition instead of restating incompatible formulas or preservation clauses. A mechanism-off or matched control is a different experiment, not an extra component of this candidate."
NEW_DEFINITION_NOTICE="\n\nUse open_mechanism.operations as the single operational definition of this candidate's changed mathematics and active parameters: specify the relevant operations, dimensions, coefficients, reductions and gradient destinations, including what is replaced and what is inherited. Keep implementation_requirements consistent with that definition as implementation obligations only, without a second algorithm. Use the other existing fields for motivation, predictions and alternative tests; refer to that definition instead of restating incompatible formulas or preservation clauses. A mechanism-off or matched control is a different experiment, not an extra component of this candidate."
CONSTRUCTION=(
 'Define one complete candidate-local mechanism. Construct from the supplied original root, '
 'or explicitly inherit one supplied executed parent through the existing research_action. '
 'For a parent revision, change only the declared source regions and inherit all others. '
 'The definition is not registered as a global primitive. Actual past outcomes remain evidence; '
 'an unmeasured mechanism has no established benefit.')

def research_projection(context):
 result=copy.deepcopy(context)
 task=result['knowledge_base']['search_space']
 task.pop('conceptual_search_space',None)
 task['construction']=CONSTRUCTION
 if 'root_relative_comparison' in task:
  previous=task['root_relative_comparison']['construction_semantics']
  marker='Native comparator_delta'
  assert marker in previous, 'Original comparator explanation changed'
  task['root_relative_comparison']['construction_semantics']=CONSTRUCTION+' '+previous[previous.index(marker):]
 return result

def definition_schema(schema):
 result=copy.deepcopy(schema)
 props=result['properties']['proposals']['items']['properties']
 props['conceptual_operator_id']={'type':'string','enum':['mechanism']}
 for key in ('conceptual_primitive_ids','conceptual_custom_components'):
  props[key]={'type':'array','items':{'type':'string'},'minItems':0,'maxItems':0}
 for key in ('conceptual_target_primitive_id','conceptual_replacement_primitive_id'):
  props[key]={'type':'null'}
 from open_mechanism import mechanism_schema
 props['open_mechanism']=mechanism_schema()
 return result

def definition_template(original):
 start=original.index('Choose exactly one definition route.')
 end=original.index('For an original-root complete new mechanism,',start)
 return original[:start]+original[end:]

@contextmanager
def discovery_scope(arm):
 if arm=='control':
  yield
  return
 if arm!='direct_definition':raise ValueError('Unknown generation arm')
 import research_actions,open_mechanism
 original_schema=research_actions.ResearchActionAdapter.proposal_schema
 original_build=open_mechanism.OpenMechanismAdapter.build_provider_research_producer
 def schema(adapter):return definition_schema(original_schema(adapter))
 def build(adapter,**kwargs):
  producer=original_build(adapter,**kwargs)
  producer._producer.proposal_template=definition_template(producer._producer.proposal_template)
  original_research_call=producer._producer._call
  def research_call(role,view,**call_options):
   return original_research_call(role,research_projection(view),**call_options)
  producer._producer._call=research_call
  original_call=producer._producer.provider_call
  def provider_call(**kwargs):
   prompt=kwargs['prompt']
   if prompt.count(OLD_DEFINITION_NOTICE)!=1:raise ValueError('Original definition notice changed')
   return original_call(**{**kwargs,'prompt':prompt.replace(OLD_DEFINITION_NOTICE,NEW_DEFINITION_NOTICE,1)})
  producer._producer.provider_call=provider_call
  return producer
 with ExitStack() as stack:
  stack.enter_context(patch.object(research_actions.ResearchActionAdapter,'proposal_schema',schema))
  stack.enter_context(patch.object(open_mechanism.OpenMechanismAdapter,'build_provider_research_producer',build))
  yield
