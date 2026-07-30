"""Negative candidate that violates the frozen pairwise input contract."""

from recbole.model.general_recommender.bpr import BPR
from recbole.utils import InputType


class InvalidInputFixtureModel(BPR):
    input_type = InputType.POINTWISE
