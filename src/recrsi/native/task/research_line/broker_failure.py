"""Keep observed schema-invalid scientific output in failure diagnostics only."""
from contextlib import contextmanager
from unittest.mock import patch
import jsonschema


@contextmanager
def preserve_schema_failures():
    from recclaw_core.experiments.helix_abc_v1.lab_api_broker import LabApiCanaryBrokerV1 as Broker
    from recclaw_core.research_line import provider
    parse, persist = Broker._parse_response, Broker._persist_failure
    require_success = provider._require_success

    def parse_response(self, **kwargs):
        self._finite_invalid_scientific_output = None
        try:
            return parse(self, **kwargs)
        except jsonschema.ValidationError as error:
            envelope = kwargs['envelope']
            # Parse reached schema validation, so original content shape has
            # already passed the broker. Never retain headers/config/keys.
            if getattr(self, 'wire_api', 'chat_completions') == 'responses':
                content = next(p['text'] for item in envelope['output']
                    if item.get('type') == 'message' for p in item['content']
                    if p.get('type') == 'output_text' and isinstance(p.get('text'),str))
            else:
                content = envelope['choices'][0]['message']['content']
                if isinstance(content,list):
                    content = ''.join(str(p.get('text') or p.get('content') or '')
                                      for p in content if isinstance(p,dict))
            def detail(e):
                return dict(instance_path=list(e.absolute_path),schema_path=list(e.absolute_schema_path),
                            validator=e.validator,children=[detail(c) for c in e.context])
            self._finite_invalid_scientific_output = dict(
                original_response_content=content,validation=detail(error),
                artifact_status='SCHEMA_INVALID_NOT_USABLE_RESPONSE',
                scientific_output_failure=True,external_transport_failure=False)
            raise

    def persist_failure(self, **kwargs):
        diagnostic = getattr(self,'_finite_invalid_scientific_output',None)
        if kwargs['error_type']=='SCHEMA_VALIDATION_FAILURE' and diagnostic is not None:
            kwargs['error_detail'] = {**kwargs['error_detail'],**diagnostic}
        try:
            return persist(self, **kwargs)
        finally:
            self._finite_invalid_scientific_output = None

    def require(result, *, kind):
        if result.call is None and (result.failure or {}).get('failure_class')=='SCHEMA_VALIDATION_FAILURE':
            raise provider.ProviderResponseRejected('Provider scientific output failed schema validation',
                provider_result=result,stage='schema_validation')
        return require_success(result,kind=kind)

    with patch.object(Broker,'_parse_response',parse_response),patch.object(Broker,'_persist_failure',persist_failure),patch.object(provider,'_require_success',require):
        yield
