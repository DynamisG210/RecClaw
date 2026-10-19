"""Prospective graph single-program role correction; no shared default change."""
from contextlib import contextmanager
from unittest.mock import patch

VERSION = 'graph-single-program-falsification-choice-v1'
INSTRUCTION = (
    'Identify the most consequential unresolved explanation using the actual '
    'measured results. Choose either a direct improvement or a discriminating '
    'experiment; a control is not required merely because this is the '
    'falsification role. When a similar control has already been measured, '
    'explain in the existing research_question and predicted_distinction fields '
    'what remains unanswered and how the proposed computation distinguishes it. '
    'A different parent name alone does not establish new information, while '
    'similar mechanism names do not establish computational equivalence. '
    'Retain useful negative results and state uncertainty; do not invent an '
    'equivalence proof or a runtime estimate. Keep protocol and dependency '
    'declarations within the exact lists below.'
)


@contextmanager
def graph_program_choice_scope():
    """Enter only in a newly versioned graph single-program composition."""
    from recclaw_core.research_line import provider
    prior = provider._RESEARCH_ROLE_INSTRUCTIONS
    expected = 'Design a mechanism around a decisive matched-control falsifier, with '
    if not prior['falsification_designer'].startswith(expected):
        raise ValueError('Review changed native role guidance before applying correction')
    revised = dict(prior, falsification_designer=INSTRUCTION)
    with patch.object(provider, '_RESEARCH_ROLE_INSTRUCTIONS', revised):
        yield
