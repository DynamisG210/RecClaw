"""Versioned decoder for the single observed AutoSpace schema-tag alias.

Original paid bytes, failed validation and scientific fields are never changed.
This is parser recovery evidence, not an independently generated valid response.
"""
import copy,hashlib,json
from pathlib import Path
from jsonschema import Draft202012Validator,ValidationError

VERSION='graph-autospace-observed-schema-tag-decoder-v1'
OBSERVED='recclaw.graph-autospace.provider-draft.source-bound.v1'
CANONICAL='recclaw.auto-space.provider-draft.source-bound.v1'

def decode(response,schema):
    errors=list(Draft202012Validator(schema).iter_errors(response))
    if not (len(errors)==1 and list(errors[0].path)==['schema'] and errors[0].validator=='const'
            and response.get('schema')==OBSERVED and schema['properties']['schema']['const']==CANONICAL):
        raise ValueError('Response is outside the one observed schema-tag recovery')
    corrected=copy.deepcopy(response);corrected['schema']=CANONICAL
    Draft202012Validator(schema).validate(corrected)
    assert {k:v for k,v in response.items() if k!='schema'}=={k:v for k,v in corrected.items() if k!='schema'}
    return corrected

def install():
    import graph_ro_generation as generation
    if getattr(generation,'_observed_schema_tag_decoder',False):return
    native_generate,native_stage=generation.generate,generation.stage
    def stage(name,request,root,settings,ledger):
        try:return native_stage(name,request,root,settings,ledger)
        except ValidationError:
            if name!='GENESIS':raise
            original=Path(root)/(name+'_RESPONSE.json')
            response=json.loads(original.read_bytes());corrected=decode(response,request['response_schema'])
            generation.freeze(Path(root)/(name+'_DECODED_RESPONSE.json'),corrected)
            generation.freeze(Path(root)/(name+'_SCHEMA_TAG_DECODER.json'),dict(version=VERSION,
                original_response_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                decoded_response_sha256=hashlib.sha256((Path(root)/(name+'_DECODED_RESPONSE.json')).read_bytes()).hexdigest(),
                changed_field='schema',observed_alias=OBSERVED,canonical_tag=CANONICAL,
                all_scientific_fields_exactly_preserved=True,original_validation_failure_retained=True,
                original_schema_validator_unchanged=True,new_model_calls=0,
                evidence_class='VERSIONED_PARSER_RECOVERY_NOT_INDEPENDENT_VALID_GENERATION'))
            return corrected
    def generate(inputs,*,stage_call=None):
        return native_generate(inputs,stage_call=stage if stage_call is None else stage_call)
    generation.generate=generate
    generation._observed_schema_tag_decoder=True
