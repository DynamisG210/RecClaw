"""Positive candidate-local model for the real RecBole qualification fixture."""

from recbole.model.general_recommender.bpr import BPR


class QualifiedFixtureModel(BPR):
    """A candidate-local entrypoint using the frozen standard BPR contract."""
