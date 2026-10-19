"""Bind V4 to the concrete adapter after original Full scopes construct it."""
from contextlib import contextmanager
from unittest.mock import patch

@contextmanager
def fresh_counterproposal_scope():
    from recclaw_core.research_line import standalone
    from counterproposal_decision import adapter_class
    native = standalone.compose_standalone_campaign
    def compose(config, **kwargs):
        adapter = kwargs['search_space_adapter']
        # No constructor/state reset: add only V4 proposal methods to this instance.
        adapter.__class__ = adapter_class(type(adapter))
        return native(config, **kwargs)
    with patch.object(standalone, 'compose_standalone_campaign', compose):
        yield
